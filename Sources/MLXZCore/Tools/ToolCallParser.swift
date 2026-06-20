import Foundation

/// A streaming parser that extracts Qwen / Hermes-style tool calls from a model's text stream.
///
/// Qwen models emit tool calls as:
/// ```
/// <tool_call>
/// {"name": "get_weather", "arguments": {"city": "Paris"}}
/// </tool_call>
/// ```
/// possibly several in a row, interleaved with ordinary assistant text. The high-level
/// mlx-swift-lm API does not parse these back out, so we own it here.
///
/// Feed text deltas in via `consume(_:)`; it returns the *visible* text to forward to the
/// client (with tool-call blocks stripped) plus any completed `ToolCall`s found. Call
/// `finish()` at end-of-stream to flush any trailing buffered text.
///
/// The parser is resilient to tag boundaries splitting across deltas: it buffers a small
/// tail that could be the start of an opening tag rather than emitting it prematurely.
public struct ToolCallParser: Sendable {
    public struct Output: Sendable, Equatable {
        public var visibleText: String
        public var toolCalls: [ToolCall]
        public init(visibleText: String = "", toolCalls: [ToolCall] = []) {
            self.visibleText = visibleText
            self.toolCalls = toolCalls
        }
    }

    private static let openTag = "<tool_call>"
    private static let closeTag = "</tool_call>"

    private enum Mode {
        case text          // outside a tool call
        case insideCall    // accumulating JSON between the tags
    }

    private var mode: Mode = .text
    /// Pending characters not yet classified (holds potential partial tags).
    private var buffer: String = ""
    /// Accumulated JSON body of the current tool call.
    private var callBody: String = ""
    private var callIndex = 0
    private let idPrefix: String

    /// - Parameter idPrefix: prefix for generated tool-call ids. Defaults to a process-unique
    ///   token so ids never collide ACROSS requests/turns — each request builds a fresh parser, so
    ///   a per-parser counter alone would emit `call_1` every turn, and Copilot matches tool
    ///   results to calls by id: duplicate ids across an agent loop break that matching and stall
    ///   the agent (it re-issues the same call and then returns an empty response).
    public init(idPrefix: String? = nil) {
        self.idPrefix = idPrefix ?? "call_\(UUID().uuidString.prefix(8))"
    }

    /// Consume a chunk of model text, returning visible text + any completed tool calls.
    public mutating func consume(_ chunk: String) -> Output {
        buffer += chunk
        var out = Output()
        process(&out, atEnd: false)
        return out
    }

    /// Flush at end of stream. Any buffered text that is not a partial tag becomes visible.
    public mutating func finish() -> Output {
        var out = Output()
        process(&out, atEnd: true)
        // At end-of-stream, whatever remains in `buffer` in text mode is real text.
        if mode == .text, !buffer.isEmpty {
            out.visibleText += buffer
            buffer = ""
        }
        return out
    }

    // MARK: - Internals

    private mutating func process(_ out: inout Output, atEnd: Bool) {
        var madeProgress = true
        while madeProgress {
            madeProgress = false
            switch mode {
            case .text:
                if let range = buffer.range(of: Self.openTag) {
                    // Emit text before the tag, then enter the call.
                    out.visibleText += String(buffer[buffer.startIndex..<range.lowerBound])
                    buffer.removeSubrange(buffer.startIndex..<range.upperBound)
                    mode = .insideCall
                    callBody = ""
                    madeProgress = true
                } else {
                    // No complete open tag. Emit everything that cannot be the start of one,
                    // keeping a tail that might be a partial "<tool_call>".
                    let safeCount = emittableTextCount(in: buffer, guarding: Self.openTag, atEnd: atEnd)
                    if safeCount > 0 {
                        let idx = buffer.index(buffer.startIndex, offsetBy: safeCount)
                        out.visibleText += String(buffer[buffer.startIndex..<idx])
                        buffer.removeSubrange(buffer.startIndex..<idx)
                    }
                }
            case .insideCall:
                if let range = buffer.range(of: Self.closeTag) {
                    callBody += String(buffer[buffer.startIndex..<range.lowerBound])
                    buffer.removeSubrange(buffer.startIndex..<range.upperBound)
                    if let call = parseCallBody(callBody) {
                        out.toolCalls.append(call)
                    }
                    callBody = ""
                    mode = .text
                    madeProgress = true
                } else {
                    // Accumulate body, but keep a tail that might be a partial "</tool_call>".
                    let safeCount = emittableTextCount(in: buffer, guarding: Self.closeTag, atEnd: atEnd)
                    if safeCount > 0 {
                        let idx = buffer.index(buffer.startIndex, offsetBy: safeCount)
                        callBody += String(buffer[buffer.startIndex..<idx])
                        buffer.removeSubrange(buffer.startIndex..<idx)
                    }
                }
            }
        }
    }

    /// Number of leading characters of `text` that are safe to emit without risking that they
    /// are the beginning of `tag`. At end-of-stream, everything is safe.
    private func emittableTextCount(in text: String, guarding tag: String, atEnd: Bool) -> Int {
        if atEnd { return text.count }
        // Find the longest suffix of `text` that is a (strict) prefix of `tag`; that suffix
        // must be retained in case the rest of the tag arrives in a later chunk.
        let maxKeep = min(text.count, tag.count - 1)
        var keep = 0
        let chars = Array(text)
        let tagChars = Array(tag)
        for len in stride(from: maxKeep, through: 1, by: -1) {
            let suffix = chars[(chars.count - len)...]
            if Array(suffix) == Array(tagChars[0..<len]) {
                keep = len
                break
            }
        }
        return text.count - keep
    }

    /// Parse a single tool-call body in either format the Qwen templates use:
    ///   - JSON:  `{"name": ..., "arguments": {...}}`
    ///   - XML :  `<function=NAME><parameter=ARG>value</parameter>...</function>`
    /// Qwen3.6's chat template instructs the XML form; older/Hermes templates use JSON.
    private mutating func parseCallBody(_ body: String) -> ToolCall? {
        let trimmed = body.trimmingCharacters(in: .whitespacesAndNewlines)
        if trimmed.contains("<function=") {
            return parseXMLCallBody(trimmed)
        }
        callIndex += 1
        // Shared name/arguments JSON parsing (also normalizes smart quotes). nil if not valid / no name.
        guard let call = Self.parseNameArgumentsJSON(trimmed, id: "\(idPrefix)_\(callIndex)") else {
            callIndex -= 1
            return nil
        }
        return call
    }

    /// Parse the Qwen XML call body: `<function=NAME>` then zero or more
    /// `<parameter=KEY>\nVALUE\n</parameter>` blocks. Values are strings as emitted; we coerce
    /// obvious JSON scalars (numbers/bools) so the arguments JSON is faithful, defaulting to string.
    private mutating func parseXMLCallBody(_ body: String) -> ToolCall? {
        guard let fnRange = body.range(of: "<function="),
            let fnEnd = body.range(of: ">", range: fnRange.upperBound ..< body.endIndex)
        else { return nil }
        let name = String(body[fnRange.upperBound ..< fnEnd.lowerBound])
            .trimmingCharacters(in: .whitespacesAndNewlines)
        guard !name.isEmpty else { return nil }

        var args: [String: Any] = [:]
        var search = fnEnd.upperBound
        while let pStart = body.range(of: "<parameter=", range: search ..< body.endIndex),
            let pNameEnd = body.range(of: ">", range: pStart.upperBound ..< body.endIndex),
            let pClose = body.range(of: "</parameter>", range: pNameEnd.upperBound ..< body.endIndex)
        {
            let key = String(body[pStart.upperBound ..< pNameEnd.lowerBound])
                .trimmingCharacters(in: .whitespacesAndNewlines)
            let raw = String(body[pNameEnd.upperBound ..< pClose.lowerBound])
                .trimmingCharacters(in: .whitespacesAndNewlines)
            if !key.isEmpty { args[key] = Self.coerceScalar(raw) }
            search = pClose.upperBound
        }

        callIndex += 1
        let argumentsJSON = args.isEmpty ? "{}" : (Self.compactJSON(args) ?? "{}")
        return ToolCall(id: "\(idPrefix)_\(callIndex)", name: name, argumentsJSON: argumentsJSON)
    }

    /// Coerce a parameter string to a JSON-faithful scalar: int, double, bool, or string fallback.
    /// (Module-internal so sibling format parsers — e.g. `GLM4OutputParser` — can reuse it.)
    static func coerceScalar(_ s: String) -> Any {
        if let i = Int(s) { return i }
        if let d = Double(s) { return d }
        if s == "true" { return true }
        if s == "false" { return false }
        return s
    }

    /// Serialize a JSON container (object or array) to compact text, or nil on failure.
    static func compactJSON(_ value: Any) -> String? {
        guard JSONSerialization.isValidJSONObject(value),
              let data = try? JSONSerialization.data(withJSONObject: value),
              let str = String(data: data, encoding: .utf8)
        else { return nil }
        return str
    }

    /// Normalize a model-emitted JSON snippet before parsing: map smart/curly quotes to straight
    /// quotes and trim surrounding whitespace. Some models (notably gpt-oss/harmony) emit `“…”`/`‘…’`
    /// in tool-call arguments, which `JSONSerialization` rejects — this is the fix for the observed
    /// `web_search` "could not parse arguments" failure. Shared by all format parsers.
    static func normalizeJSONText(_ s: String) -> String {
        s.replacingOccurrences(of: "\u{201C}", with: "\"")   // “ left double
         .replacingOccurrences(of: "\u{201D}", with: "\"")   // ” right double
         .replacingOccurrences(of: "\u{201E}", with: "\"")   // „ low double
         .replacingOccurrences(of: "\u{2018}", with: "'")    // ‘ left single
         .replacingOccurrences(of: "\u{2019}", with: "'")    // ’ right single
         .trimmingCharacters(in: .whitespacesAndNewlines)
    }

    /// Parse a JSON tool-call body of the canonical `{"name": ..., "arguments": {...}}` shape into a
    /// `ToolCall`, applying `normalizeJSONText` first. Returns the compact arguments JSON exactly as
    /// `parseCallBody` does (string args passed through, object/array compacted, else `{}`). Shared so
    /// harmony and other formats produce identical argument serialization. Returns nil if not valid
    /// JSON or missing a name.
    static func parseNameArgumentsJSON(_ body: String, id: String) -> ToolCall? {
        let trimmed = normalizeJSONText(body)
        guard let data = trimmed.data(using: .utf8),
              let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let name = obj["name"] as? String
        else { return nil }
        return ToolCall(id: id, name: name, argumentsJSON: argumentsJSONString(from: obj["arguments"]))
    }

    /// Build the compact arguments-JSON string from a decoded `arguments` value, matching the rules in
    /// `parseCallBody`: a pre-serialized string passes through; an object/array is compacted; anything
    /// else (missing/null/scalar) becomes `{}`.
    static func argumentsJSONString(from arguments: Any?) -> String {
        switch arguments {
        case let str as String: return str
        case let container as [String: Any]: return compactJSON(container) ?? "{}"
        case let array as [Any]: return compactJSON(array) ?? "{}"
        default: return "{}"
        }
    }
}
