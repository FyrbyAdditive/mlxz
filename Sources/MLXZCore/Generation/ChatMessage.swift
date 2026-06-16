import Foundation

/// A single message in a conversation, normalized away from any wire format.
public struct ChatMessage: Sendable, Equatable {
    public enum Role: String, Sendable, Codable {
        case system
        case user
        case assistant
        case tool
    }

    public var role: Role
    public var content: [ContentPart]

    /// Tool calls emitted by an assistant turn (when replaying history back to the model).
    public var toolCalls: [ToolCall]

    /// For `.tool` messages: the id of the tool call this is responding to.
    public var toolCallID: String?

    public init(
        role: Role,
        content: [ContentPart],
        toolCalls: [ToolCall] = [],
        toolCallID: String? = nil
    ) {
        self.role = role
        self.content = content
        self.toolCalls = toolCalls
        self.toolCallID = toolCallID
    }

    /// Convenience for a plain-text message.
    public init(role: Role, text: String) {
        self.init(role: role, content: [.text(text)])
    }

    /// All text content concatenated (ignores non-text parts).
    public var text: String {
        content.compactMap { part in
            if case let .text(t) = part { return t }
            return nil
        }.joined()
    }

    /// True if any content part carries an image.
    public var hasImages: Bool {
        content.contains { part in
            switch part {
            case .imageURL, .imageData: return true
            case .text: return false
            }
        }
    }
}

/// A piece of message content. Extensible for future modalities.
public enum ContentPart: Sendable, Equatable {
    case text(String)
    case imageURL(URL)
    case imageData(Data)
}
