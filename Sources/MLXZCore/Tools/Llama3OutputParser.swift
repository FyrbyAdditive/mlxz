import Foundation

/// `OutputParser` for Llama 3+ inline tool calls. Llama emits either `<|python_tag|>{json}` or bare
/// inline JSON `{"name":…, "parameters"|"arguments":…}` with no wrapper tags. No reasoning channel, so
/// non-tool text streams as visible. Mirrors the fork's `Llama3ToolCallParser` (JSON form; the rarer
/// pythonic form is out of scope — non-JSON falls through as visible text).
public struct Llama3OutputParser: OutputParser {
    private static let pythonTag = "<|python_tag|>"

    private enum Mode { case text, collecting }
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
        case .collecting:
            if !buffer.isEmpty {
                if let c = parseCall(buffer) { out.toolCalls.append(c) }
                else { out.visibleText += buffer }   // wasn't a tool call → show it
                buffer = ""
            }
        }
        return out
    }

    private mutating func process(_ out: inout OutputParse, atEnd: Bool) {
        var progress = true
        while progress {
            progress = false
            switch mode {
            case .text:
                if let r = buffer.range(of: Self.pythonTag) {
                    // Explicit python tag unambiguously begins a tool call.
                    out.visibleText += String(buffer[buffer.startIndex ..< r.lowerBound])
                    buffer.removeSubrange(buffer.startIndex ..< r.upperBound)
                    mode = .collecting
                    progress = true
                } else if let brace = buffer.firstIndex(of: "{") {
                    // Bare inline JSON: switch to collecting at the first `{`.
                    out.visibleText += String(buffer[buffer.startIndex ..< brace])
                    buffer.removeSubrange(buffer.startIndex ..< brace)
                    mode = .collecting
                    progress = true
                } else {
                    let keep = partialSuffix(buffer, guarding: Self.pythonTag, atEnd: atEnd)
                    if keep < buffer.count {
                        let idx = buffer.index(buffer.endIndex, offsetBy: -keep)
                        out.visibleText += String(buffer[buffer.startIndex ..< idx])
                        buffer.removeSubrange(buffer.startIndex ..< idx)
                    }
                }
            case .collecting:
                // Once braces balance, decide: valid call → emit; not a call → flush as visible.
                if bracesBalanced(buffer) {
                    if let c = parseCall(buffer) { out.toolCalls.append(c) }
                    else { out.visibleText += buffer }
                    buffer = ""
                    mode = .text
                    progress = true
                }
            }
        }
    }

    private mutating func parseCall(_ body: String) -> ToolCall? {
        let normalized = ToolCallParser.normalizeJSONText(body)
        guard let data = normalized.data(using: .utf8),
              let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let name = obj["name"] as? String
        else { return nil }
        // Llama uses `parameters` (or `arguments`) for the args object.
        let args = obj["arguments"] ?? obj["parameters"]
        callIndex += 1
        return ToolCall(
            id: "\(idPrefix)_\(callIndex)", name: name,
            argumentsJSON: ToolCallParser.argumentsJSONString(from: args))
    }

    private func bracesBalanced(_ s: String) -> Bool {
        var depth = 0; var seen = false
        for ch in s {
            if ch == "{" { depth += 1; seen = true } else if ch == "}" { depth -= 1 }
        }
        return seen && depth == 0
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
