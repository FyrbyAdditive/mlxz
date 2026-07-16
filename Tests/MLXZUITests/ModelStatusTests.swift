import Testing
import Foundation
@testable import MLXZUI
@testable import MLXZHub
@testable import MLXZCore

@Suite struct ModelStatusTests {

    // MARK: - Unified status precedence

    @Test func loadedWinsOverEverything() {
        let s = ModelStatus.resolve(
            loaded: true, loading: true, isDrafter: true, attachedDrafter: true,
            downloadFraction: 0.5, downloadFailed: true, installed: true)
        #expect(s == .loaded)
    }

    @Test func loadingWinsOverDownloadAndInstalled() {
        let s = ModelStatus.resolve(
            loaded: false, loading: true, isDrafter: false, attachedDrafter: false,
            downloadFraction: 0.5, downloadFailed: false, installed: true)
        #expect(s == .loading)
    }

    @Test func drafterReportedWithAttachment() {
        #expect(ModelStatus.resolve(
            loaded: false, loading: false, isDrafter: true, attachedDrafter: true,
            downloadFraction: nil, downloadFailed: false, installed: true) == .drafter(attached: true))
        #expect(ModelStatus.resolve(
            loaded: false, loading: false, isDrafter: true, attachedDrafter: false,
            downloadFraction: nil, downloadFailed: false, installed: true) == .drafter(attached: false))
    }

    @Test func downloadingBeatsInstalled() {
        let s = ModelStatus.resolve(
            loaded: false, loading: false, isDrafter: false, attachedDrafter: false,
            downloadFraction: 0.3, downloadFailed: false, installed: true)
        #expect(s == .downloading(fraction: 0.3))
    }

    @Test func installedShowsLoadNotDownload() {
        // The bug the redesign fixes: an already-installed search hit must NOT show Download.
        let s = ModelStatus.resolve(
            loaded: false, loading: false, isDrafter: false, attachedDrafter: false,
            downloadFraction: nil, downloadFailed: false, installed: true)
        #expect(s == .installed)
    }

    @Test func remoteWhenNothingKnown() {
        let s = ModelStatus.resolve(
            loaded: false, loading: false, isDrafter: false, attachedDrafter: false,
            downloadFraction: nil, downloadFailed: false, installed: false)
        #expect(s == .remote)
    }

    @Test func failedDownloadWhenNotInstalled() {
        let s = ModelStatus.resolve(
            loaded: false, loading: false, isDrafter: false, attachedDrafter: false,
            downloadFraction: nil, downloadFailed: true, installed: false)
        #expect(s == .downloadFailed)
    }

    // MARK: - RAM fit

    @Test func ramFitThresholds() {
        let ram: UInt64 = 16 * 1_000_000_000            // 16 GB
        #expect(RAMFit.of(sizeBytes: 8_000_000_000, physicalMemory: ram) == .fits)      // 50%
        #expect(RAMFit.of(sizeBytes: 11_000_000_000, physicalMemory: ram) == .tight)    // 69%
        #expect(RAMFit.of(sizeBytes: 14_000_000_000, physicalMemory: ram) == .exceeds)  // 88%
        #expect(RAMFit.of(sizeBytes: nil, physicalMemory: ram) == .unknown)
        #expect(RAMFit.of(sizeBytes: 0, physicalMemory: ram) == .unknown)
    }

    // MARK: - Discover filtering

    @Test func hidesDrafters() {
        let entries = [
            entry("mlx-community/Qwen3-8B-4bit"),
            entry("mlx-community/Qwen3.6-27B-MTP-4bit"),   // MTP drafter → hidden
        ]
        let visible = DiscoverFilter.hidingDrafters(entries).map(\.id)
        #expect(visible == ["mlx-community/Qwen3-8B-4bit"])
    }

    @Test func facetFiltering() {
        let vlm = entry("mlx-community/Qwen2.5-VL-7B-Instruct-4bit", tags: ["mlx", "image-text-to-text"])
        let coder = entry("mlx-community/Qwen2.5-Coder-7B-4bit")
        let chat = entry("mlx-community/Qwen3-8B-Instruct-4bit")
        let all = [vlm, coder, chat]

        #expect(DiscoverFilter.apply(all, facet: .vision).map(\.id) == [vlm.id])
        #expect(DiscoverFilter.apply(all, facet: .code).map(\.id) == [coder.id])
        #expect(DiscoverFilter.apply(all, facet: .all).count == 3)
        // Chat excludes vision and code.
        let chatIDs = DiscoverFilter.apply(all, facet: .chat).map(\.id)
        #expect(chatIDs.contains(chat.id))
        #expect(!chatIDs.contains(vlm.id))
        #expect(!chatIDs.contains(coder.id))
    }

    // MARK: - Formatting

    @Test func compactCounts() {
        #expect(CountFormat.compact(742) == "742")
        #expect(CountFormat.compact(1_500) == "1.5k")
        #expect(CountFormat.compact(142_000) == "142k")
        #expect(CountFormat.compact(1_200_000) == "1.2M")
    }

    private func entry(_ id: String, tags: [String] = ["mlx"]) -> CatalogEntry {
        CatalogEntry(id: id, downloads: 0, likes: 0, tags: tags, lastModified: nil, sizeBytes: nil)
    }
}
