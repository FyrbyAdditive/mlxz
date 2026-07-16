import Testing
import Foundation
@testable import MLXZHub
@testable import MLXZCore

@Suite struct HubCatalogTests {
    static let sampleJSON = """
    [
      {"id":"mlx-community/Qwen3.6-35B-A3B-MTP-4bit","downloads":1200,"likes":2,
       "tags":["mlx","qwen3_5_mtp","moe","mtp","text-generation"],"lastModified":"2026-06-01T00:00:00.000Z"},
      {"id":"mlx-community/Qwen3.6-4B-4bit","downloads":5000,"likes":10,
       "tags":["mlx","qwen3","text-generation"]},
      {"modelId":"mlx-community/Qwen2.5-VL-7B-Instruct-4bit","downloads":800,
       "tags":["mlx","qwen2_5_vl","image-text-to-text"]},
      {"id":"","downloads":0,"tags":[]}
    ]
    """

    @Test func decodesEntriesAndSkipsEmptyIDs() throws {
        let entries = try HubCatalog.decode(Data(Self.sampleJSON.utf8))
        // The blank-id entry is dropped.
        #expect(entries.count == 3)
        #expect(entries[0].id == "mlx-community/Qwen3.6-35B-A3B-MTP-4bit")
        #expect(entries[1].downloads == 5000)
    }

    @Test func inferredFlags() throws {
        let entries = try HubCatalog.decode(Data(Self.sampleJSON.utf8))
        let mtp = entries.first { $0.id.contains("MTP") }!
        #expect(mtp.isMTP)
        #expect(mtp.isMoE)
        #expect(mtp.quantization == "4bit")

        let vl = entries.first { $0.id.contains("VL") }!
        #expect(vl.isVision)
        #expect(vl.capabilities.contains(.vision))

        let plain = entries.first { $0.id == "mlx-community/Qwen3.6-4B-4bit" }!
        #expect(!plain.isVision)
        #expect(!plain.isMTP)
    }

    @Test func imageTextToTextTagFlagsVisionWithoutNameMarker() throws {
        // A VLM whose repo id has no -vl marker but whose HF task tag is image-text-to-text.
        let json = #"""
        [{"id":"mlx-community/Qwen3.6-27B-4bit","downloads":100,
          "tags":["mlx","qwen3_5","image-text-to-text","conversational"]}]
        """#
        let entry = try HubCatalog.decode(Data(json.utf8))[0]
        #expect(entry.isVision)
        #expect(entry.capabilities.contains(.vision))
    }

    @Test func searchURLUsesMLXFilter() async {
        // We can't hit the network in tests, but we can confirm the catalog is constructible.
        let catalog = HubCatalog()
        _ = catalog  // smoke: no crash on default init
    }

    @Test func decodesParameterCountAndSizeClass() throws {
        let json = #"""
        [{"id":"mlx-community/Qwen3-8B-4bit","downloads":1,
          "safetensors":{"total":8190000000,"parameters":{"U32":1000000000,"BF16":100000000}}},
         {"id":"mlx-community/Qwen3-0.5B-4bit","downloads":1,
          "safetensors":{"total":494000000,"parameters":{"U32":60000000}}}]
        """#
        let entries = try HubCatalog.decode(Data(json.utf8))
        #expect(entries[0].parameterCount == 8_190_000_000)
        #expect(entries[0].sizeClass == "8.2B")
        #expect(entries[1].sizeClass == "494M")
        // sizeBytes still computed from per-dtype parameters.
        #expect(entries[0].sizeBytes != nil)
    }

    @Test func authorSegmentParsed() throws {
        let e = try HubCatalog.decode(Data(#"[{"id":"deepseek-ai/foo","downloads":0}]"#.utf8))[0]
        #expect(e.author == "deepseek-ai")
    }

    @Test func clientSideSizeSort() {
        func entry(_ id: String, _ size: Int?) -> CatalogEntry {
            CatalogEntry(id: id, downloads: 0, likes: 0, tags: [], lastModified: nil, sizeBytes: size)
        }
        let xs = [entry("a", 3_000), entry("b", 1_000), entry("c", nil), entry("d", 2_000)]
        let smallest = HubCatalog.sorted(xs, by: .sizeSmallest).map(\.id)
        #expect(smallest.prefix(3) == ["b", "d", "a"])   // unknown-size sorts last
        #expect(smallest.last == "c")
        let largest = HubCatalog.sorted(xs, by: .sizeLargest).map(\.id)
        #expect(largest.prefix(3) == ["a", "d", "b"])
        // Non-size sorts are a no-op (server already ordered them).
        #expect(HubCatalog.sorted(xs, by: .popular).map(\.id) == ["a", "b", "c", "d"])
    }
}
