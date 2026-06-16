import Foundation

/// A tool the model may call, normalized from the OpenAI `function` shape.
public struct ToolDefinition: Sendable, Equatable {
    public var name: String
    public var description: String?
    /// JSON Schema for the parameters object, as raw JSON text.
    public var parametersJSONSchema: String?

    public init(name: String, description: String? = nil, parametersJSONSchema: String? = nil) {
        self.name = name
        self.description = description
        self.parametersJSONSchema = parametersJSONSchema
    }
}

/// A concrete tool invocation produced by the model.
public struct ToolCall: Sendable, Equatable, Identifiable {
    public var id: String
    public var name: String
    /// The arguments object as raw JSON text (what gets passed back to OpenAI clients verbatim).
    public var argumentsJSON: String

    public init(id: String, name: String, argumentsJSON: String) {
        self.id = id
        self.name = name
        self.argumentsJSON = argumentsJSON
    }
}
