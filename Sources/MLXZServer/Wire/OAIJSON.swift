import Foundation

/// A tiny JSON value builder with *insertion-ordered* object keys, so serialized output is
/// deterministic and unit-testable (unlike a `[String: Any]` + JSONSerialization, which
/// orders keys arbitrarily).
indirect enum OAIJSON: Sendable {
    case null
    case bool(Bool)
    case int(Int)
    case double(Double)
    case string(String)
    case array([OAIJSON])
    /// Ordered key/value pairs.
    case object([(String, OAIJSON)])

    /// Convenience for dictionary-literal-ish construction at call sites.
    static func object(_ pairs: KeyValuePairs<String, OAIJSON>) -> OAIJSON {
        .object(pairs.map { ($0.key, $0.value) })
    }

    func serialized() throws -> Data {
        var s = ""
        write(into: &s)
        guard let data = s.data(using: .utf8) else {
            throw APIErrorBox.encoding
        }
        return data
    }

    var jsonString: String {
        var s = ""
        write(into: &s)
        return s
    }

    private func write(into s: inout String) {
        switch self {
        case .null:
            s += "null"
        case .bool(let b):
            s += b ? "true" : "false"
        case .int(let i):
            s += String(i)
        case .double(let d):
            s += String(d)
        case .string(let str):
            s += Self.encodeString(str)
        case .array(let arr):
            s += "["
            for (i, el) in arr.enumerated() {
                if i > 0 { s += "," }
                el.write(into: &s)
            }
            s += "]"
        case .object(let pairs):
            s += "{"
            for (i, pair) in pairs.enumerated() {
                if i > 0 { s += "," }
                s += Self.encodeString(pair.0)
                s += ":"
                pair.1.write(into: &s)
            }
            s += "}"
        }
    }

    /// JSON string escaping per RFC 8259.
    private static func encodeString(_ str: String) -> String {
        var out = "\""
        for scalar in str.unicodeScalars {
            switch scalar {
            case "\"": out += "\\\""
            case "\\": out += "\\\\"
            case "\n": out += "\\n"
            case "\r": out += "\\r"
            case "\t": out += "\\t"
            case let s where s.value < 0x20:
                out += String(format: "\\u%04x", s.value)
            default:
                out.unicodeScalars.append(scalar)
            }
        }
        out += "\""
        return out
    }
}

private enum APIErrorBox: Error { case encoding }

/// Id and timestamp helpers for OpenAI-shaped responses.
enum OpenAIID {
    static func random() -> String {
        // 24 lowercase-alphanumeric chars, matching OpenAI's id style.
        let chars = "abcdefghijklmnopqrstuvwxyz0123456789"
        var s = ""
        for _ in 0..<24 {
            s.append(chars.randomElement()!)
        }
        return s
    }

    static func timestamp() -> Int {
        Int(Date().timeIntervalSince1970)
    }
}
