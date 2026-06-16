import Foundation
import NIOCore
import MLXZCore

/// One Server-Sent-Events frame.
struct SSEFrame: Equatable {
    /// Optional `event:` name (used by the Responses API; nil for chat completions).
    var event: String?
    /// The `data:` payload (already-serialized JSON, or the literal `[DONE]`).
    var data: String

    /// Serialize to wire bytes: `event: <name>\n` (optional) + `data: <data>\n\n`.
    var wireText: String {
        var s = ""
        if let event { s += "event: \(event)\n" }
        s += "data: \(data)\n\n"
        return s
    }

    func byteBuffer(allocator: ByteBufferAllocator) -> ByteBuffer {
        allocator.buffer(string: wireText)
    }
}

/// Maps the internal `GenerationEvent` stream onto one OpenAI streaming dialect.
/// Implementations are stateful (they accumulate ids, indices, etc.) but are only ever
/// touched serially from within a single response-body writer task, hence `Sendable`.
protocol SSEEventEncoder: AnyObject, Sendable {
    func encode(_ event: GenerationEvent) throws -> [SSEFrame]
    /// Final frames after the event stream finishes (e.g. chat completions emits `[DONE]`).
    func terminator() -> [SSEFrame]
}
