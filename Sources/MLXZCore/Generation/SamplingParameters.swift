import Foundation

/// Sampling controls, independent of any wire format.
public struct SamplingParameters: Sendable, Equatable {
    public var temperature: Float
    public var topP: Float
    public var topK: Int?
    public var repetitionPenalty: Float?
    public var seed: UInt64?

    public init(
        temperature: Float = 0.7,
        topP: Float = 1.0,
        topK: Int? = nil,
        repetitionPenalty: Float? = nil,
        seed: UInt64? = nil
    ) {
        self.temperature = temperature
        self.topP = topP
        self.topK = topK
        self.repetitionPenalty = repetitionPenalty
        self.seed = seed
    }

    public static let `default` = SamplingParameters()
}

/// Speculative-decoding configuration. The seam for MTP / draft-model decoding.
public struct SpeculativeConfig: Sendable, Equatable {
    public enum Mode: Sendable, Equatable {
        /// Use the model's built-in multi-token-prediction heads (e.g. `qwen3_5_mtp`).
        case mtp
        /// Use a separate small draft model identified by repo id.
        case draftModel(modelID: String)
        /// Opt this request out of ALL speculative decoding (plain decode). Used by the
        /// losslessness benchmark to compare speculative vs plain output in one process.
        case disabled
    }

    public var mode: Mode
    public var numDraftTokens: Int

    public init(mode: Mode, numDraftTokens: Int = 4) {
        self.mode = mode
        self.numDraftTokens = numDraftTokens
    }
}
