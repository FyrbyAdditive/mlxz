import Foundation
import Observation

/// A bounded, observable ring buffer of human-readable log lines surfaced in the UI.
@MainActor
@Observable
public final class LogStore {
    public struct Line: Identifiable, Sendable {
        public let id = UUID()
        public let text: String
    }

    public private(set) var lines: [Line] = []
    private let capacity: Int

    public init(capacity: Int = 500) {
        self.capacity = capacity
    }

    public func append(_ text: String) {
        lines.append(Line(text: text))
        if lines.count > capacity {
            lines.removeFirst(lines.count - capacity)
        }
    }

    public func clear() {
        lines.removeAll()
    }
}
