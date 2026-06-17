import Foundation

/// Watches system memory pressure and invokes a handler on warning/critical events.
/// Used to auto-unload the model before the OS kills the app under pressure (large MoE models).
final class MemoryPressureMonitor: @unchecked Sendable {
    private let source: DispatchSourceMemoryPressure
    private let handler: @Sendable (Bool) -> Void   // isCritical

    /// - Parameter onPressure: called with `true` for critical, `false` for warning.
    init(onPressure: @escaping @Sendable (Bool) -> Void) {
        self.handler = onPressure
        self.source = DispatchSource.makeMemoryPressureSource(eventMask: [.warning, .critical], queue: .global())
        source.setEventHandler { [source, handler] in
            handler(source.data.contains(.critical))
        }
        source.resume()
    }

    deinit {
        source.cancel()
    }
}
