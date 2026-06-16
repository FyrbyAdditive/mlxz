import Foundation

/// What a loaded model supports. Drives request validation and what we advertise to clients.
public struct ModelCapabilities: OptionSet, Sendable {
    public let rawValue: Int
    public init(rawValue: Int) { self.rawValue = rawValue }

    public static let chat        = ModelCapabilities(rawValue: 1 << 0)
    public static let tools       = ModelCapabilities(rawValue: 1 << 1)
    public static let vision      = ModelCapabilities(rawValue: 1 << 2)
    public static let embeddings  = ModelCapabilities(rawValue: 1 << 3)
    public static let speculative = ModelCapabilities(rawValue: 1 << 4)
}
