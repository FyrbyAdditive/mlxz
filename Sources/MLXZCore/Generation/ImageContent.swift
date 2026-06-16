import Foundation

/// Parses an OpenAI image reference (an `image_url` string) into a `ContentPart`.
/// Handles both remote URLs and base64 `data:` URLs, which clients use to inline images.
public enum ImageContent {
    public static func part(fromURLString s: String) -> ContentPart? {
        if s.hasPrefix("data:") {
            // data:[<mediatype>][;base64],<data>
            guard let comma = s.firstIndex(of: ",") else { return nil }
            let meta = s[s.index(s.startIndex, offsetBy: 5)..<comma]
            let payload = String(s[s.index(after: comma)...])
            if meta.contains("base64") {
                guard let data = Data(base64Encoded: payload) else { return nil }
                return .imageData(data)
            } else if let decoded = payload.removingPercentEncoding, let data = decoded.data(using: .utf8) {
                return .imageData(data)
            }
            return nil
        }
        guard let url = URL(string: s) else { return nil }
        return .imageURL(url)
    }
}
