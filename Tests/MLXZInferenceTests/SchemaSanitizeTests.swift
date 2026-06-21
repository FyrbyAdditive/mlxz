import Foundation
import Testing

@testable import MLXZInference

/// `MLXInferenceEngine.sanitizeSchema` must guarantee every property node has a string `type`, so
/// chat templates that do `value['type'] | upper` (Gemma) never hit a non-string and crash with
/// "upper filter requires string". Covers the cases seen from real clients: missing type, union type,
/// and an empty `{}` property (the `make_chart` tool's `x` point).
@Suite struct SchemaSanitizeTests {
    private func type(of node: Any?) -> String? { (node as? [String: any Sendable])?["type"] as? String }

    @Test func emptyPropertySchemaGetsStringType() {
        // `{ "x": {} }` — the exact make_chart shape that crashed Gemma.
        let schema: [String: any Sendable] = [
            "type": "object",
            "properties": ["x": [String: any Sendable]()],
        ]
        let out = MLXInferenceEngine.sanitizeSchema(schema)
        let props = out["properties"] as? [String: any Sendable]
        #expect(type(of: props?["x"]) == "string")
    }

    @Test func missingTypeDefaultsToString() {
        let schema: [String: any Sendable] = [
            "type": "object",
            "properties": ["q": ["description": "a query"] as [String: any Sendable]],
        ]
        let out = MLXInferenceEngine.sanitizeSchema(schema)
        let props = out["properties"] as? [String: any Sendable]
        #expect(type(of: props?["q"]) == "string")
    }

    @Test func unionTypeCollapsesToFirstString() {
        let schema: [String: any Sendable] = [
            "type": "object",
            "properties": ["x": ["type": ["string", "null"]] as [String: any Sendable]],
        ]
        let out = MLXInferenceEngine.sanitizeSchema(schema)
        let props = out["properties"] as? [String: any Sendable]
        #expect(type(of: props?["x"]) == "string")
    }

    @Test func nestedArrayItemsAreSanitized() {
        // series[].points[].x == {} must be fixed deep in the tree.
        let point: [String: any Sendable] = ["properties": ["x": [String: any Sendable](), "y": ["type": "number"] as [String: any Sendable]]]
        let points: [String: any Sendable] = ["type": "array", "items": point]
        let seriesItem: [String: any Sendable] = ["type": "object", "properties": ["points": points]]
        let schema: [String: any Sendable] = [
            "type": "object",
            "properties": ["series": ["type": "array", "items": seriesItem] as [String: any Sendable]],
        ]
        let out = MLXInferenceEngine.sanitizeSchema(schema)
        // Walk down to x.
        let series = (out["properties"] as? [String: any Sendable])?["series"] as? [String: any Sendable]
        let si = series?["items"] as? [String: any Sendable]
        let pts = (si?["properties"] as? [String: any Sendable])?["points"] as? [String: any Sendable]
        let pt = pts?["items"] as? [String: any Sendable]
        let x = (pt?["properties"] as? [String: any Sendable])?["x"]
        #expect(type(of: x) == "string")
    }

    @Test func validSchemaPassesThrough() {
        let schema: [String: any Sendable] = [
            "type": "object",
            "properties": ["q": ["type": "string", "description": "d"] as [String: any Sendable]],
            "required": ["q"],
        ]
        let out = MLXInferenceEngine.sanitizeSchema(schema)
        #expect(out["type"] as? String == "object")
        let props = out["properties"] as? [String: any Sendable]
        #expect(type(of: props?["q"]) == "string")
        #expect((out["required"] as? [String]) == ["q"])
    }
}
