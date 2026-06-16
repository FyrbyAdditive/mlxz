import Foundation
import MLXZCore

/// Generates the VS Code Copilot "Custom Endpoint" model declaration for a loaded model,
/// so a user can copy-paste it into their `chatLanguageModels.json`. This is the key UX that
/// makes the primary consumer (Copilot BYOK) trivial to wire up.
public enum CopilotConfig {
    /// Build the `chatLanguageModels.json` entry for one model served by this app.
    ///
    /// - Parameters:
    ///   - repoID: the model id (sent as `id` and shown as `name`).
    ///   - host/port: where the server is bound.
    ///   - capabilities: drives `toolCalling` / `vision`.
    ///   - apiType: "chat-completions" (default) or "responses".
    ///   - maxInputTokens/maxOutputTokens: advertised context window.
    public static func modelEntry(
        repoID: String,
        host: String = "127.0.0.1",
        port: Int = 8080,
        capabilities: ModelCapabilities,
        apiType: String = "chat-completions",
        maxInputTokens: Int = 128_000,
        maxOutputTokens: Int = 8_192
    ) -> String {
        let url = "http://\(host):\(port)/v1"
        let displayName = repoID.split(separator: "/").last.map(String.init) ?? repoID
        let lines = [
            "{",
            "  \"id\": \(jsonString(repoID)),",
            "  \"name\": \(jsonString("mlxz · \(displayName)")),",
            "  \"url\": \(jsonString(url)),",
            "  \"apiType\": \(jsonString(apiType)),",
            "  \"toolCalling\": \(capabilities.contains(.tools)),",
            "  \"vision\": \(capabilities.contains(.vision)),",
            "  \"maxInputTokens\": \(maxInputTokens),",
            "  \"maxOutputTokens\": \(maxOutputTokens)",
            "}",
        ]
        return lines.joined(separator: "\n")
    }

    /// A short human-readable setup hint shown alongside the snippet.
    public static func setupHint(host: String = "127.0.0.1", port: Int = 8080) -> String {
        """
        In VS Code, add a Custom (OpenAI-compatible) endpoint pointing at \
        http://\(host):\(port)/v1, then paste the model entry below into your \
        chatLanguageModels.json. Tool calling and streaming are supported.
        """
    }

    private static func jsonString(_ s: String) -> String {
        let escaped = s
            .replacingOccurrences(of: "\\", with: "\\\\")
            .replacingOccurrences(of: "\"", with: "\\\"")
        return "\"\(escaped)\""
    }
}
