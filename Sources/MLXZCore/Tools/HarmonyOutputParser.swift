import Foundation

/// `OutputParser` for the OpenAI **"harmony"** response format used by gpt-oss.
///
/// Harmony encodes everything as channel blocks delimited by special markers. A block looks like:
/// ```
/// [<|start|>assistant[ to=functions.NAME]]<|channel|>CHANNEL[ to=functions.NAME][ <|constrain|>json]<|message|>CONTENT<TERMINAL>
/// ```
/// where:
///   - `CHANNEL` = `analysis` → reasoning, `final` → visible answer, `commentary` → a tool call;
///   - a tool call carries `to=functions.NAME` (after `<|start|>` or after `<|channel|>commentary`)
///     and its `CONTENT` is the JSON arguments object;
///   - `TERMINAL` ∈ `<|end|>` (block end), `<|return|>` (assistant EOS), `<|call|>` (end of a tool call).
///
/// Example (one observed turn):
/// ```
/// <|channel|>analysis<|message|>We should search.<|end|><|start|>assistant<|channel|>commentary
/// to=functions.web_search <|constrain|>json<|message|>{"query":"x","max_results":10}<|call|>
/// ```
///
/// Streaming contract: `analysis`/`final` content streams incrementally (routed to reasoning/visible);
/// `commentary` (tool) content is accumulated whole, then parsed into a `ToolCall` on its terminal.
/// Markers split across chunk boundaries are buffered and never leaked. The parser is defensive: it
/// treats `<|return|>` / `<|call|>` as block-terminal even if generation didn't physically stop, so the
/// emitted channels stay correct regardless of stop-token configuration.
public struct HarmonyOutputParser: OutputParser {
    // MARK: Markers
    private static let mChannel = "<|channel|>"
    private static let mMessage = "<|message|>"
    private static let mStart = "<|start|>"
    private static let mEnd = "<|end|>"
    private static let mReturn = "<|return|>"
    private static let mCall = "<|call|>"
    private static let mConstrain = "<|constrain|>"
    /// All markers, used to compute the partial-suffix tail to retain across chunks.
    private static let allMarkers = [mChannel, mMessage, mStart, mEnd, mReturn, mCall, mConstrain]
    private static let maxMarkerLen = allMarkers.map(\.count).max() ?? 0
    /// Markers that terminate a block's content (route + close).
    private static let terminals = [mEnd, mReturn, mCall]

    // MARK: State
    private enum Channel { case analysis, final, commentary, unknown }
    private enum Mode {
        case header        // between blocks / reading a header up to <|message|>
        case body(Channel) // streaming/accumulating content up to a terminal
    }

    private var mode: Mode = .header
    private var buffer = ""                 // unclassified text (may hold a partial marker)
    private var commentaryBuffer = ""        // accumulated tool-call JSON body
    /// Tool name captured from the most recent `to=functions.NAME` (on `<|start|>` or `<|channel|>`).
    private var pendingToolName: String?

    private var callIndex = 0
    private let idPrefix: String

    /// - Parameter idPrefix: tool-call id prefix (defaults to a process-unique token, matching
    ///   `ToolCallParser` — ids must be unique across requests so clients can match results to calls).
    public init(idPrefix: String? = nil) {
        self.idPrefix = idPrefix ?? "call_\(UUID().uuidString.prefix(8))"
    }

    public mutating func consume(_ chunk: String) -> OutputParse {
        buffer += chunk
        var out = OutputParse()
        process(&out, atEnd: false)
        return out
    }

    public mutating func finish() -> OutputParse {
        var out = OutputParse()
        process(&out, atEnd: true)
        // Flush whatever remains per the current block kind (defensive: unterminated block at EOS).
        if !buffer.isEmpty {
            routeContent(buffer, into: &out, atEnd: true)
            buffer = ""
        }
        closeBlock(into: &out)
        return out
    }

    // MARK: - Core

    private mutating func process(_ out: inout OutputParse, atEnd: Bool) {
        var progress = true
        while progress {
            progress = false
            switch mode {
            case .header:
                // Seek the start of a content block: `<|message|>` opens the body once we know the
                // channel. Until then, parse/skip header markers (`<|start|>`, `<|channel|>`,
                // `<|constrain|>`, names, `to=…`).
                if let r = buffer.range(of: Self.mMessage) {
                    // The header text precedes <|message|>; classify the channel + tool name from it.
                    let header = String(buffer[buffer.startIndex ..< r.lowerBound])
                    classifyHeader(header)
                    buffer.removeSubrange(buffer.startIndex ..< r.upperBound)
                    if case .body(let ch) = mode, ch == .commentary { commentaryBuffer = "" }
                    progress = true
                } else {
                    // No <|message|> yet. If there's clearly no header marker pending and we're at end,
                    // drop the buffer (stray non-content). Otherwise keep buffering (header continues).
                    if atEnd { buffer = "" }
                }

            case .body(let channel):
                // Find the earliest terminal marker in the buffer.
                if let term = earliestTerminal(in: buffer) {
                    let content = String(buffer[buffer.startIndex ..< term.range.lowerBound])
                    routeContent(content, into: &out, atEnd: true)
                    buffer.removeSubrange(buffer.startIndex ..< term.range.upperBound)
                    closeBlock(into: &out)
                    // `<|return|>` / `<|call|>` end the block; `<|end|>` too. After any terminal we go
                    // back to seeking the next header. (A following `<|start|>`/`<|channel|>` re-enters.)
                    mode = .header
                    progress = true
                } else if channel == .commentary {
                    // Tool-call body: accumulate everything except a possible partial terminal tail.
                    let keep = partialMarkerSuffix(buffer, atEnd: atEnd)
                    if keep < buffer.count {
                        let idx = buffer.index(buffer.endIndex, offsetBy: -keep)
                        commentaryBuffer += String(buffer[buffer.startIndex ..< idx])
                        buffer.removeSubrange(buffer.startIndex ..< idx)
                    }
                } else {
                    // analysis/final: stream content, holding only a partial-marker tail.
                    let keep = partialMarkerSuffix(buffer, atEnd: atEnd)
                    if keep < buffer.count {
                        let idx = buffer.index(buffer.endIndex, offsetBy: -keep)
                        let emit = String(buffer[buffer.startIndex ..< idx])
                        routeContent(emit, into: &out, atEnd: false)
                        buffer.removeSubrange(buffer.startIndex ..< idx)
                    }
                }
            }
        }
    }

    /// Classify the header preceding `<|message|>` into a channel + optional tool name. The header may
    /// contain `<|start|>assistant`, `<|channel|>NAME`, `to=functions.NAME`, and `<|constrain|>json`.
    private mutating func classifyHeader(_ header: String) {
        // Capture a tool target if present (after `to=`), e.g. `to=functions.web_search`.
        if let name = Self.toolName(in: header) { pendingToolName = name }

        // The channel name is the token right after `<|channel|>`.
        var channel: Channel = .unknown
        if let cr = header.range(of: Self.mChannel) {
            let after = header[cr.upperBound...]
            let token = after.prefix { !$0.isWhitespace && $0 != "<" }
                .trimmingCharacters(in: .whitespaces)
            switch token.lowercased() {
            case "analysis": channel = .analysis
            case "final": channel = .final
            case "commentary": channel = .commentary
            default: channel = .unknown
            }
        }
        mode = .body(channel)
    }

    /// Route content to the right channel of `out`. analysis→reasoning, final→visible, commentary is
    /// accumulated into `commentaryBuffer` (parsed on close); unknown is dropped.
    private mutating func routeContent(_ text: String, into out: inout OutputParse, atEnd: Bool) {
        guard !text.isEmpty else { return }
        switch mode {
        case .body(.analysis): out.reasoning += text
        case .body(.final): out.visibleText += text
        case .body(.commentary): commentaryBuffer += text
        case .body(.unknown): break   // ignore content of channels we don't surface
        case .header: break
        }
    }

    /// Close the current block: if it was a tool call, parse the accumulated JSON into a `ToolCall`.
    private mutating func closeBlock(into out: inout OutputParse) {
        if case .body(.commentary) = mode {
            let body = commentaryBuffer.trimmingCharacters(in: .whitespacesAndNewlines)
            commentaryBuffer = ""
            if !body.isEmpty {
                callIndex += 1
                let id = "\(idPrefix)_\(callIndex)"
                if let name = pendingToolName,
                   let args = Self.toolCall(name: name, argumentsBody: body, id: id) {
                    out.toolCalls.append(args)
                } else if let parsed = ToolCallParser.parseNameArgumentsJSON(body, id: id) {
                    // Fallback: body itself is `{"name":…,"arguments":…}` (no separate target seen).
                    out.toolCalls.append(parsed)
                } else {
                    callIndex -= 1
                }
            }
        }
        pendingToolName = nil
    }

    // MARK: - Helpers

    /// Build a ToolCall from a harmony tool target (name) + a JSON arguments body. The body is the
    /// arguments object directly (harmony) — normalize smart quotes, validate it's a JSON object.
    private static func toolCall(name: String, argumentsBody: String, id: String) -> ToolCall? {
        let normalized = ToolCallParser.normalizeJSONText(argumentsBody)
        // Accept a JSON object (the common case) or an empty body → {}.
        if normalized.isEmpty { return ToolCall(id: id, name: name, argumentsJSON: "{}") }
        guard let data = normalized.data(using: .utf8),
              let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any]
        else { return nil }
        let argsJSON = ToolCallParser.compactJSON(obj) ?? "{}"
        return ToolCall(id: id, name: name, argumentsJSON: argsJSON)
    }

    /// Extract `NAME` from a `to=functions.NAME` (or `to=NAME`) occurrence in `header`.
    private static func toolName(in header: String) -> String? {
        guard let r = header.range(of: "to=") else { return nil }
        var rest = header[r.upperBound...].drop { $0 == " " }
        // Optional `functions.` namespace prefix.
        if rest.hasPrefix("functions.") { rest = rest.dropFirst("functions.".count) }
        let name = rest.prefix { !$0.isWhitespace && $0 != "<" }
            .trimmingCharacters(in: .whitespaces)
        return name.isEmpty ? nil : name
    }

    /// The earliest terminal marker (`<|end|>`/`<|return|>`/`<|call|>`) in `text`, if any.
    private func earliestTerminal(in text: String) -> (range: Range<String.Index>, marker: String)? {
        var best: (range: Range<String.Index>, marker: String)?
        for m in Self.terminals {
            if let r = text.range(of: m) {
                if best == nil || r.lowerBound < best!.range.lowerBound { best = (r, m) }
            }
        }
        return best
    }

    /// Length of the trailing suffix of `text` to retain because it might be the start of a marker
    /// arriving in a later chunk. Zero at end-of-stream. (Same idea as `ToolCallParser`/`ThinkParser`.)
    private func partialMarkerSuffix(_ text: String, atEnd: Bool) -> Int {
        if atEnd { return 0 }
        let chars = Array(text)
        var keep = 0
        let maxKeep = min(chars.count, Self.maxMarkerLen - 1)
        // A retained suffix only matters if it could begin a marker — every harmony marker starts "<|".
        for len in stride(from: maxKeep, through: 1, by: -1) {
            let suffix = String(chars[(chars.count - len)...])
            if Self.allMarkers.contains(where: { $0.hasPrefix(suffix) }) {
                keep = len
                break
            }
        }
        return keep
    }
}
