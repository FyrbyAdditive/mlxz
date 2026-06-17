import Testing
@testable import MLXZCore

@Suite struct DrafterPairingTests {
    @Test func detectsDrafters() {
        #expect(DrafterPairing.isDrafter("mlx-community/Qwen3.6-27B-MTP-4bit"))
        #expect(DrafterPairing.isDrafter("mlx-community/Qwen3.5-4B-MTP-bf16"))
        #expect(!DrafterPairing.isDrafter("mlx-community/Qwen3.6-27B-4bit"))
        #expect(!DrafterPairing.isDrafter("mlx-community/Qwen2.5-0.5B-Instruct-4bit"))
    }

    @Test func baseFromDrafter() {
        #expect(
            DrafterPairing.baseRepoID(forDrafter: "mlx-community/Qwen3.6-27B-MTP-4bit")
                == "mlx-community/Qwen3.6-27B-4bit")
        #expect(
            DrafterPairing.baseRepoID(forDrafter: "mlx-community/Qwen3.5-9B-MTP-5bit")
                == "mlx-community/Qwen3.5-9B-5bit")
        #expect(DrafterPairing.baseRepoID(forDrafter: "mlx-community/Qwen3.6-27B-4bit") == nil)
    }

    @Test func drafterFromBase() {
        #expect(
            DrafterPairing.drafterRepoID(forBase: "mlx-community/Qwen3.6-27B-4bit")
                == "mlx-community/Qwen3.6-27B-MTP-4bit")
        #expect(
            DrafterPairing.drafterRepoID(forBase: "mlx-community/Qwen3.6-35B-A3B-5bit")
                == "mlx-community/Qwen3.6-35B-A3B-MTP-5bit")
    }

    @Test func roundTrips() {
        let base = "mlx-community/Qwen3.6-27B-4bit"
        let drafter = DrafterPairing.drafterRepoID(forBase: base)
        #expect(DrafterPairing.baseRepoID(forDrafter: drafter) == base)
    }
}
