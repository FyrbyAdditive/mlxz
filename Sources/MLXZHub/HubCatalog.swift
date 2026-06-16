import Foundation
import MLXZCore

/// A model entry returned from the HuggingFace Hub search API.
public struct CatalogEntry: Sendable, Identifiable, Hashable {
    public var id: String          // repo id, e.g. "mlx-community/Qwen3.6-4B-4bit"
    public var downloads: Int
    public var likes: Int
    public var tags: [String]
    public var lastModified: String?

    public var displayName: String { id.split(separator: "/").last.map(String.init) ?? id }

    /// Capabilities inferred from the repo id + tags.
    public var capabilities: ModelCapabilities {
        ModelCapabilityDetector.detect(repoID: id, modelType: tags.first { $0.contains("_") })
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
                lastModified: m.lastModified
            )
        }.filter { !$0.id.isEmpty }
    }

    /// The subset of fields the Hub API returns that we use.
    struct HubModel: Decodable {
        var id: String?
        var modelId: String?
        var downloads: Int?
        var likes: Int?
        var tags: [String]?
        var lastModified: String?
    }
}
