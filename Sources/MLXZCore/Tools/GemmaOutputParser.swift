import Foundation

/// `OutputParser` for Gemma's output format (Gemma-4 MLX checkpoints).
///
/// Tool calls: `<|tool_call>call:NAME{key:<|"|>string value<|"|>,key2:42}<tool_call|>` (possibly
/// several). String argument values are wrapped in `<|"|>…<|"|>`; bare values (numbers/bools) are
/// coerced to JSON scalars.
///
/// Reasoning: Gemma marks a channel with a `<|channel>NAME<channel|>` header; text after it (until the
/// next channel header / tool call / end) belongs to that channel. The `thought` channel is the model's
/// chain-of-thought — it's routed to `reasoning` (surfaced separately, e.g. as `reasoning_content`),
/// not dumped into the chat as the raw `<|channel>thought<channel|>` markers it used to leak. Any other
/// channel's text is treated as visible. (Mirrors the grammar of the template's `strip_thinking` macro,
/// but preserves the thought as reasoning instead of discarding it.)
public struct GemmaOutputParser: OutputParser {
    private static let callOpen = "<|tool_call>"
    private static let callClose = "<tool_call|>"
    private static let strMarker = "<|\"|>"
    private static let chanOpen = "<|channel>"
    private static let chanClose = "<channel|>"
    /// Markers that can begin in `.text` mode; we hold a partial-suffix tail covering the longest.
    private static let textMarkers = [callOpen, chanOpen]
    private static let maxTextMarkerLen = textMarkers.map(\.count).max() ?? 0

    private enum Mode { case text, insideCall, insideChannelHeader }
    /// Which channel the current `.text` content belongs to (nil = visible; "thought" = reasoning).
    private enum Channel { case visible, reasoning }

    private var mode: Mode = .text
    private var channel: Channel = .visible
    private var buffer = ""
    private var callBody = ""
    private var headerBuffer = ""
    private var callIndex = 0
    private let idPrefix: String

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
        if mode == .text, !buffer.isEmpty { emitText(buffer, into: &out); buffer = "" }
        // Unterminated call at EOS: best-effort parse what we have.
        if mode == .insideCall, !callBody.isEmpty, let c = parseCall(callBody) {
            out.toolCalls.append(c); callBody = ""
        }
        return out
    }

    private mutating func process(_ out: inout OutputParse, atEnd: Bool) {
        var progress = true
        while progress {
            progress = false
            switch mode {
            case .text:
                // Find the earliest of a tool-call open or a channel header.
                let call = buffer.range(of: Self.callOpen)
                let chan = buffer.range(of: Self.chanOpen)
                let firstCall = call?.lowerBound
                let firstChan = chan?.lowerBound
                if let c = call, firstCall != nil, (firstChan == nil || firstCall! <= firstChan!) {
                    emitText(String(buffer[buffer.startIndex ..< c.lowerBound]), into: &out)
                    buffer.removeSubrange(buffer.startIndex ..< c.upperBound)
                    mode = .insideCall; callBody = ""
                    progress = true
                } else if let h = chan {
                    emitText(String(buffer[buffer.startIndex ..< h.lowerBound]), into: &out)
                    buffer.removeSubrange(buffer.startIndex ..< h.upperBound)
                    mode = .insideChannelHeader; headerBuffer = ""
                    progress = true
                } else {
                    let keep = partialSuffixAny(buffer, markers: Self.textMarkers,
                                                maxLen: Self.maxTextMarkerLen, atEnd: atEnd)
                    if keep < buffer.count {
                        let idx = buffer.index(buffer.endIndex, offsetBy: -keep)
                        emitText(String(buffer[buffer.startIndex ..< idx]), into: &out)
                        buffer.removeSubrange(buffer.startIndex ..< idx)
                    }
                }

            case .insideChannelHeader:
                // The channel name runs from `<|channel>` up to `<channel|>`.
                if let r = buffer.range(of: Self.chanClose) {
                    headerBuffer += String(buffer[buffer.startIndex ..< r.lowerBound])
                    buffer.removeSubrange(buffer.startIndex ..< r.upperBound)
                    let name = headerBuffer.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
                    channel = (name == "thought") ? .reasoning : .visible
                    headerBuffer = ""; mode = .text
                    progress = true
                } else {
                    let keep = partialSuffix(buffer, guarding: Self.chanClose, atEnd: atEnd)
                    if keep < buffer.count {
                        let idx = buffer.index(buffer.endIndex, offsetBy: -keep)
                        headerBuffer += String(buffer[buffer.startIndex ..< idx])
                        buffer.removeSubrange(buffer.startIndex ..< idx)
                    }
                }

            case .insideCall:
                if let r = buffer.range(of: Self.callClose) {
                    callBody += String(buffer[buffer.startIndex ..< r.lowerBound])
                    buffer.removeSubrange(buffer.startIndex ..< r.upperBound)
                    if let c = parseCall(callBody) { out.toolCalls.append(c) }
                    callBody = ""; mode = .text
                    progress = true
                } else {
                    let keep = partialSuffix(buffer, guarding: Self.callClose, atEnd: atEnd)
                    if keep < buffer.count {
                        let idx = buffer.index(buffer.endIndex, offsetBy: -keep)
                        callBody += String(buffer[buffer.startIndex ..< idx])
                        buffer.removeSubrange(buffer.startIndex ..< idx)
                    }
                }
            }
        }
    }

    /// Route `.text`-mode content to the active channel (visible or reasoning).
    private func emitText(_ text: String, into out: inout OutputParse) {
        guard !text.isEmpty else { return }
        switch channel {
        case .visible: out.visibleText += text
        case .reasoning: out.reasoning += text
        }
    }

    /// Parse `call:NAME{key:value,...}` where string values are wrapped in `<|"|>…<|"|>`.
    private mutating func parseCall(_ body: String) -> ToolCall? {
        let text = body.trimmingCharacters(in: .whitespacesAndNewlines)
        guard let callRange = text.range(of: "call:") else { return nil }
        let rest = text[callRange.upperBound...]
        guard let braceStart = rest.firstIndex(of: "{") else { return nil }
        let name = String(rest[..<braceStart]).trimmingCharacters(in: .whitespacesAndNewlines)
        guard !name.isEmpty else { return nil }
        guard let braceEnd = rest.lastIndex(of: "}") else { return nil }
        let argsStr = String(rest[rest.index(after: braceStart) ..< braceEnd])

        let args = parseArgs(argsStr)
        callIndex += 1
        let argsJSON = args.isEmpty ? "{}" : (ToolCallParser.compactJSON(args) ?? "{}")
        return ToolCall(id: "\(idPrefix)_\(callIndex)", name: name, argumentsJSON: argsJSON)
    }

    /// Parse the comma-separated `key:value` argument list. Values wrapped in `<|"|>…<|"|>` are
    /// strings (and may legitimately contain commas); bare values are coerced to JSON scalars.
    private func parseArgs(_ s: String) -> [String: Any] {
        var args: [String: Any] = [:]
        var rest = Substring(s)
        while !rest.isEmpty {
            guard let colon = rest.firstIndex(of: ":") else { break }
            let key = rest[..<colon].trimmingCharacters(in: .whitespacesAndNewlines)
            rest = rest[rest.index(after: colon)...]
            while let f = rest.first, f == " " { rest = rest.dropFirst() }

            if rest.hasPrefix(Self.strMarker) {
                let afterOpen = rest.dropFirst(Self.strMarker.count)
                if let close = afterOpen.range(of: Self.strMarker) {
                    let value = String(afterOpen[..<close.lowerBound])
                    if !key.isEmpty { args[key] = value }
                    rest = afterOpen[close.upperBound...]
                } else {
                    if !key.isEmpty { args[key] = String(afterOpen) }
                    rest = ""
                }
            } else {
                let commaIdx = rest.firstIndex(of: ",") ?? rest.endIndex
                let raw = rest[..<commaIdx].trimmingCharacters(in: .whitespacesAndNewlines)
                if !key.isEmpty { args[key] = ToolCallParser.coerceScalar(raw) }
                rest = commaIdx < rest.endIndex ? rest[rest.index(after: commaIdx)...] : ""
            }
            while let f = rest.first, f == " " || f == "," { rest = rest.dropFirst() }
        }
        return args
    }

    /// Length of the trailing suffix of `text` to keep because it might begin `tag` in a later chunk.
    private func partialSuffix(_ text: String, guarding tag: String, atEnd: Bool) -> Int {
        if atEnd { return 0 }
        let chars = Array(text); let tagChars = Array(tag)
        let maxKeep = min(chars.count, tagChars.count - 1)
        for len in stride(from: maxKeep, through: 1, by: -1) {
            if Array(chars[(chars.count - len)...]) == Array(tagChars[0 ..< len]) { return len }
        }
        return 0
    }

    /// Like `partialSuffix` but for any of several markers (keeps the longest matching tail).
    private func partialSuffixAny(_ text: String, markers: [String], maxLen: Int, atEnd: Bool) -> Int {
        if atEnd { return 0 }
        let chars = Array(text)
        let maxKeep = min(chars.count, maxLen - 1)
        for len in stride(from: maxKeep, through: 1, by: -1) {
            let suffix = String(chars[(chars.count - len)...])
            if markers.contains(where: { $0.hasPrefix(suffix) }) { return len }
        }
        return 0
    }
}
