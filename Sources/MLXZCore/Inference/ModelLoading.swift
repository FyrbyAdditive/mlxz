import Foundation

/// Progress of a model load (which may include an on-demand download).
public struct LoadProgress: Sendable {
    /// 0.0 ... 1.0, or nil if indeterminate.
    public var fraction: Double?
    public var detail: String?

    public init(fraction: Double? = nil, detail: String? = nil) {
        self.fraction = fraction
        self.detail = detail
    }
}

/// Loads a model descriptor into a ready-to-use `InferenceEngine`.
///
/// This is the seam that keeps `ModelManager` (and the whole app) free of any MLX dependency:
/// the MLX-backed loader is injected at the composition root.
public protocol ModelLoading: Sendable {
    func load(
        _ descriptor: ModelDescriptor,
        progress: @escaping @Sendable (LoadProgress) -> Void
    ) async throws -> any InferenceEngine
}
