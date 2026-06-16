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

    public init(idPrefix: String = "call") {
        self.idPrefix = idPrefix
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

    /// Parse a single tool-call JSON body of the form `{"name": ..., "arguments": {...}}`.
    private mutating func parseCallBody(_ body: String) -> ToolCall? {
        let trimmed = body.trimmingCharacters(in: .whitespacesAndNewlines)
        guard let data = trimmed.data(using: .utf8),
              let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let name = obj["name"] as? String
        else { return nil }

        let argumentsJSON: String
        switch obj["arguments"] {
        case let str as String:
            // Some templates emit arguments pre-serialized as a JSON string; pass through.
            argumentsJSON = str
        case let container as [String: Any]:
            argumentsJSON = Self.compactJSON(container) ?? "{}"
        case let array as [Any]:
            argumentsJSON = Self.compactJSON(array) ?? "{}"
        default:
            // Missing, null, or a scalar (which JSONSerialization can't serialize at top level).
            argumentsJSON = "{}"
        }

        callIndex += 1
        return ToolCall(id: "\(idPrefix)_\(callIndex)", name: name, argumentsJSON: argumentsJSON)
    }

    /// Serialize a JSON container (object or array) to compact text, or nil on failure.
    private static func compactJSON(_ value: Any) -> String? {
        guard JSONSerialization.isValidJSONObject(value),
              let data = try? JSONSerialization.data(withJSONObject: value),
              let str = String(data: data, encoding: .utf8)
        else { return nil }
        return str
    }
}
