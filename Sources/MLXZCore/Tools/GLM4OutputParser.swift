import Foundation

/// `OutputParser` for GLM4 tool calls: `<tool_call>name<arg_key>k</arg_key><arg_value>v</arg_value>…
/// </tool_call>`. Same outer `<tool_call>` wrapper as Qwen, but a key/value body instead of JSON. No
/// reasoning channel (non-tool text streams as visible). Mirrors the fork's `GLM4ToolCallParser`.
public struct GLM4OutputParser: OutputParser {
    private static let openTag = "<tool_call>"
    private static let closeTag = "</tool_call>"

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

    /// Parse `name<arg_key>k</arg_key><arg_value>v</arg_value>…` into a ToolCall. Values are coerced to
    /// JSON scalars where they parse as such (GLM4 emits raw values; we default to string).
    private mutating func parseCall(_ body: String) -> ToolCall? {
        let text = body.trimmingCharacters(in: .whitespacesAndNewlines)
        let keyStart = text.range(of: "<arg_key>")
        let name = String(text[..<(keyStart?.lowerBound ?? text.endIndex)])
            .trimmingCharacters(in: .whitespacesAndNewlines)
        guard !name.isEmpty else { return nil }

        var args: [String: Any] = [:]
        var search = text.startIndex
        while let ks = text.range(of: "<arg_key>", range: search ..< text.endIndex),
              let ke = text.range(of: "</arg_key>", range: ks.upperBound ..< text.endIndex),
              let vs = text.range(of: "<arg_value>", range: ke.upperBound ..< text.endIndex),
              let ve = text.range(of: "</arg_value>", range: vs.upperBound ..< text.endIndex) {
            let key = String(text[ks.upperBound ..< ke.lowerBound]).trimmingCharacters(in: .whitespacesAndNewlines)
            let val = String(text[vs.upperBound ..< ve.lowerBound]).trimmingCharacters(in: .whitespacesAndNewlines)
            if !key.isEmpty { args[key] = ToolCallParser.coerceScalar(val) }
            search = ve.upperBound
        }
        callIndex += 1
        let argsJSON = args.isEmpty ? "{}" : (ToolCallParser.compactJSON(args) ?? "{}")
        return ToolCall(id: "\(idPrefix)_\(callIndex)", name: name, argumentsJSON: argsJSON)
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
