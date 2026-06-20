import Foundation

/// `OutputParser` for Mistral V11+ tool calls: `[TOOL_CALLS]name[ARGS]{json}` (repeated for multiple
/// calls). There is no end tag — a call runs to the next `[TOOL_CALLS]` or to end-of-stream, so the
/// body is accumulated and parsed when the next call starts or at `finish()`. Mistral has no separate
/// reasoning channel, so non-tool text streams as visible. Mirrors the fork's `MistralToolCallParser`.
public struct MistralOutputParser: OutputParser {
    private static let startTag = "[TOOL_CALLS]"
    private static let argsTag = "[ARGS]"
    private static let callIdTag = "[CALL_ID]"

    private enum Mode { case text, insideCall }
    private var mode: Mode = .text
    private var buffer = ""        // single buffer; meaning depends on `mode`
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
        switch mode {
        case .text:
            if !buffer.isEmpty { out.visibleText += buffer; buffer = "" }
        case .insideCall:
            // No end tag — the trailing call is finalized here.
            if !buffer.isEmpty, let c = parseCall(buffer) { out.toolCalls.append(c) }
            buffer = ""
        }
        return out
    }

    private mutating func process(_ out: inout OutputParse, atEnd: Bool) {
        var progress = true
        while progress {
            progress = false
            switch mode {
            case .text:
                if let r = buffer.range(of: Self.startTag) {
                    out.visibleText += String(buffer[buffer.startIndex ..< r.lowerBound])
                    buffer.removeSubrange(buffer.startIndex ..< r.upperBound)
                    mode = .insideCall
                    progress = true
                } else {
                    let keep = partialSuffix(buffer, guarding: Self.startTag, atEnd: atEnd)
                    if keep < buffer.count {
                        let idx = buffer.index(buffer.endIndex, offsetBy: -keep)
                        out.visibleText += String(buffer[buffer.startIndex ..< idx])
                        buffer.removeSubrange(buffer.startIndex ..< idx)
                    }
                }
            case .insideCall:
                // A subsequent [TOOL_CALLS] ends this call and begins another: parse, re-enter.
                if let r = buffer.range(of: Self.startTag) {
                    let one = String(buffer[buffer.startIndex ..< r.lowerBound])
                    if let c = parseCall(one) { out.toolCalls.append(c) }
                    buffer.removeSubrange(buffer.startIndex ..< r.upperBound)
                    progress = true
                }
                // Otherwise keep accumulating in `buffer`; finalized at finish() (no end tag).
            }
        }
    }

    /// Parse one `name[ARGS]{json}` (optionally `name[CALL_ID]…[ARGS]{json}`) body.
    private mutating func parseCall(_ body: String) -> ToolCall? {
        let text = body.trimmingCharacters(in: .whitespacesAndNewlines)
        guard let argsRange = text.range(of: Self.argsTag) else { return nil }
        var namePart = String(text[..<argsRange.lowerBound]).trimmingCharacters(in: .whitespacesAndNewlines)
        if let cid = namePart.range(of: Self.callIdTag) {
            namePart = String(namePart[..<cid.lowerBound]).trimmingCharacters(in: .whitespacesAndNewlines)
        }
        let argsPart = String(text[argsRange.upperBound...]).trimmingCharacters(in: .whitespacesAndNewlines)
        guard !namePart.isEmpty else { return nil }
        let normalized = ToolCallParser.normalizeJSONText(argsPart)
        guard let data = normalized.data(using: .utf8),
              let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any]
        else { return nil }
        callIndex += 1
        return ToolCall(
            id: "\(idPrefix)_\(callIndex)", name: namePart,
            argumentsJSON: ToolCallParser.compactJSON(obj) ?? "{}")
    }

    private func partialSuffix(_ text: String, guarding tag: String, atEnd: Bool) -> Int {
        if atEnd { return 0 }
        let chars = Array(text); let tagChars = Array(tag)
        let maxKeep = min(chars.count, tagChars.count - 1)
        for len in stride(from: maxKeep, through: 1, by: -1) {
            if Array(chars[(chars.count - len)...]) == Array(tagChars[0 ..< len]) { return len }
        }
        return 0
    }
}
