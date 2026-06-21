import Foundation

/// `OutputParser` for Gemma's function-call format. Gemma-4 (MLX) emits tool calls as:
/// ```
/// <|tool_call>call:NAME{key:<|"|>string value<|"|>,key2:42}<tool_call|>
/// ```
/// possibly several in a row. String argument values are wrapped in `<|"|>…<|"|>`; non-string values
/// (numbers/bools) appear bare. Gemma has no separate reasoning channel, so non-tool text streams as
/// visible. (The fork's `GemmaFunctionParser` targets a different, tag-based variant
/// `<start_function_call>…call:name{…}…<end_function_call>` with `<escape>` strings; this matches the
/// `<|tool_call>`/`<tool_call|>`/`<|"|>` markers actually produced by the Gemma-4 MLX checkpoints.)
public struct GemmaOutputParser: OutputParser {
    private static let openTag = "<|tool_call>"
    private static let closeTag = "<tool_call|>"
    private static let strMarker = "<|\"|>"

    private enum Mode { case text, insideCall }
    private var mode: Mode = .text
    private var buffer = ""
    private var callBody = ""
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
        if mode == .text, !buffer.isEmpty { out.visibleText += buffer; buffer = "" }
        // An unterminated call at EOS: best-effort parse what we have.
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
                if let r = buffer.range(of: Self.openTag) {
                    out.visibleText += String(buffer[buffer.startIndex ..< r.lowerBound])
                    buffer.removeSubrange(buffer.startIndex ..< r.upperBound)
                    mode = .insideCall; callBody = ""
                    progress = true
                } else {
                    let keep = partialSuffix(buffer, guarding: Self.openTag, atEnd: atEnd)
                    if keep < buffer.count {
                        let idx = buffer.index(buffer.endIndex, offsetBy: -keep)
                        out.visibleText += String(buffer[buffer.startIndex ..< idx])
                        buffer.removeSubrange(buffer.startIndex ..< idx)
                    }
                }
            case .insideCall:
                if let r = buffer.range(of: Self.closeTag) {
                    callBody += String(buffer[buffer.startIndex ..< r.lowerBound])
                    buffer.removeSubrange(buffer.startIndex ..< r.upperBound)
                    if let c = parseCall(callBody) { out.toolCalls.append(c) }
                    callBody = ""; mode = .text
                    progress = true
                } else {
                    let keep = partialSuffix(buffer, guarding: Self.closeTag, atEnd: atEnd)
                    if keep < buffer.count {
                        let idx = buffer.index(buffer.endIndex, offsetBy: -keep)
                        callBody += String(buffer[buffer.startIndex ..< idx])
                        buffer.removeSubrange(buffer.startIndex ..< idx)
                    }
                }
            }
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
            // Leading whitespace before the value.
            while let f = rest.first, f == " " { rest = rest.dropFirst() }

            if rest.hasPrefix(Self.strMarker) {
                // Quoted string: <|"|>…<|"|> — find the closing marker (commas inside are literal).
                let afterOpen = rest.dropFirst(Self.strMarker.count)
                if let close = afterOpen.range(of: Self.strMarker) {
                    let value = String(afterOpen[..<close.lowerBound])
                    if !key.isEmpty { args[key] = value }
                    rest = afterOpen[close.upperBound...]
                } else {
                    // Unterminated string marker — take the remainder.
                    if !key.isEmpty { args[key] = String(afterOpen) }
                    rest = ""
                }
            } else {
                // Bare value up to the next comma.
                let commaIdx = rest.firstIndex(of: ",") ?? rest.endIndex
                let raw = rest[..<commaIdx].trimmingCharacters(in: .whitespacesAndNewlines)
                if !key.isEmpty { args[key] = ToolCallParser.coerceScalar(raw) }
                rest = commaIdx < rest.endIndex ? rest[rest.index(after: commaIdx)...] : ""
            }
            // Skip a separating comma (and surrounding spaces).
            while let f = rest.first, f == " " || f == "," { rest = rest.dropFirst() }
        }
        return args
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
