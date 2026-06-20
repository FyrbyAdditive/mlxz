import Foundation
import MLXZCore

/// A model entry returned from the HuggingFace Hub search API.
public struct CatalogEntry: Sendable, Identifiable, Hashable {
    public var id: String          // repo id, e.g. "mlx-community/Qwen3.6-4B-4bit"
    public var downloads: Int
    public var likes: Int
    public var tags: [String]
    public var lastModified: String?
    /// Approximate on-disk download size in bytes, computed from the Hub's per-dtype safetensors
    /// parameter counts (exact for MLX repos: U32-packed 4-bit weights + BF16 scales/norms). nil if
    /// the Hub didn't report safetensors metadata for the repo.
    public var sizeBytes: Int?

    public var displayName: String { id.split(separator: "/").last.map(String.init) ?? id }

    /// Human-readable download size, e.g. "16.1 GB" / "240 MB", or nil if unknown.
    public var sizeString: String? {
        guard let b = sizeBytes, b > 0 else { return nil }
        let gb = Double(b) / 1_000_000_000
        if gb >= 1 { return String(format: "%.1f GB", gb) }
        return String(format: "%.0f MB", Double(b) / 1_000_000)
    }

    /// Capabilities inferred from the repo id + tags. The HF task tag `image-text-to-text`
    /// (or a `vision`/`multimodal` tag) marks a VLM even when the repo id has no `-vl` marker.
    public var capabilities: ModelCapabilities {
        let visionTags: Set<String> = ["image-text-to-text", "image-to-text", "vision", "multimodal"]
        let hasVisionTag = tags.contains { visionTags.contains($0.lowercased()) }
        return ModelCapabilityDetector.detect(
            repoID: id, modelType: tags.first { $0.contains("_") }, hasVisionConfig: hasVisionTag)
    }

    public var isMoE: Bool { id.localizedCaseInsensitiveContains("a3b") || tags.contains("moe") || id.lowercased().contains("moe") }
    public var isMTP: Bool { capabilities.contains(.speculative) }
    public var isVision: Bool { capabilities.contains(.vision) }

    /// Quantization label parsed from the id (e.g. "4bit", "5bit", "8bit", "bf16").
    public var quantization: String? {
        let lower = id.lowercased()
        for q in ["2bit", "3bit", "4bit", "5bit", "6bit", "8bit", "bf16", "fp16"] where lower.contains(q) {
            return q
        }
        return nil
    }
}

/// Searches the HuggingFace Hub for MLX-compatible models.
public struct HubCatalog: Sendable {
    private let session: URLSession
    private let endpoint: URL

    public init(session: URLSession = .shared, endpoint: URL = URL(string: "https://huggingface.co")!) {
        self.session = session
        self.endpoint = endpoint
    }

    /// Search for MLX models. An empty query returns trending MLX models.
    /// Defaults to the `mlx-community` org, which hosts the curated MLX conversions.
    public func search(
        query: String,
        author: String? = "mlx-community",
        limit: Int = 30
    ) async throws -> [CatalogEntry] {
        var components = URLComponents(url: endpoint.appendingPathComponent("api/models"), resolvingAgainstBaseURL: false)!
        var items = [
            URLQueryItem(name: "filter", value: "mlx"),
            URLQueryItem(name: "limit", value: String(limit)),
            URLQueryItem(name: "sort", value: "downloads"),
            URLQueryItem(name: "direction", value: "-1"),
        ]
        if !query.isEmpty { items.append(URLQueryItem(name: "search", value: query)) }
        if let author { items.append(URLQueryItem(name: "author", value: author)) }
        // Pull per-dtype safetensors parameter counts so we can show the download size in the same
        // request (no per-model follow-up). `expand[]` is the only way the list endpoint returns it.
        items.append(URLQueryItem(name: "expand[]", value: "safetensors"))
        components.queryItems = items

        let (data, response) = try await session.data(from: components.url!)
        guard let http = response as? HTTPURLResponse, (200..<300).contains(http.statusCode) else {
            throw APIError(kind: .server, message: "HuggingFace search failed", code: "hub_search_failed")
        }
        return try Self.decode(data)
    }

    /// Decode the Hub `/api/models` JSON array into catalog entries. Exposed for testing.
    public static func decode(_ data: Data) throws -> [CatalogEntry] {
        let raw = try JSONDecoder().decode([HubModel].self, from: data)
        return raw.map { m in
            CatalogEntry(
                id: m.id ?? m.modelId ?? "",
                downloads: m.downloads ?? 0,
                likes: m.likes ?? 0,
                tags: m.tags ?? [],
                lastModified: m.lastModified,
                sizeBytes: sizeBytes(m.safetensors?.parameters)
            )
        }.filter { !$0.id.isEmpty }
    }

    /// Bytes per element for the dtypes the Hub reports in `safetensors.parameters`. MLX 4-bit
    /// weights are packed as `U32` (8 nibbles per word, so the count is already per-element after
    /// unpacking → 4 bytes each as stored); scales/biases/norms are `BF16`. Unknown dtypes default to
    /// 2 bytes (conservative). Validated exact vs the local 27B-4bit (16.05 GB on disk == computed).
    static func sizeBytes(_ parameters: [String: Int]?) -> Int? {
        guard let parameters, !parameters.isEmpty else { return nil }
        let bytesPer: [String: Int] = [
            "F64": 8, "I64": 8, "U64": 8,
            "F32": 4, "I32": 4, "U32": 4,
            "BF16": 2, "F16": 2, "FP16": 2, "I16": 2, "U16": 2,
            "F8_E4M3": 1, "F8_E5M2": 1, "I8": 1, "U8": 1, "BOOL": 1,
        ]
        return parameters.reduce(0) { $0 + $1.value * (bytesPer[$1.key.uppercased()] ?? 2) }
    }

    /// The subset of fields the Hub API returns that we use.
    struct HubModel: Decodable {
        var id: String?
        var modelId: String?
        var downloads: Int?
        var likes: Int?
        var tags: [String]?
        var lastModified: String?
        var safetensors: SafeTensors?
        struct SafeTensors: Decodable {
            var total: Int?
            var parameters: [String: Int]?
        }
    }
}
