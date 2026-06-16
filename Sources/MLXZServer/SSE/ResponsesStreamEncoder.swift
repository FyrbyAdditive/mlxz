import Foundation
import MLXZCore

/// Encodes the internal event stream as OpenAI Responses-API structured SSE events.
///
/// Emits the documented sequence for a single text message output:
///   response.created → response.output_item.added → response.content_part.added →
///   response.output_text.delta* → response.output_text.done → response.content_part.done →
///   response.output_item.done → response.completed
///
/// Function calls are emitted as a `function_call` output item with
/// response.function_call_arguments.delta/done events. Each event carries `type` (inside data)
/// and a monotonically increasing `sequence_number`, matching the wire format clients parse.
final class ResponsesStreamEncoder: SSEEventEncoder, @unchecked Sendable {
    private let responseID: String
    private let itemID: String
    private let modelID: String
    private let created: Int

    private var sequence = 0
    private var openedTextItem = false
    private var accumulatedText = ""
    private var emittedItems = 0       // output_index counter across message + function items
    private var finishReason: FinishReason = .stop
    private var usage: TokenUsage = .init()
    private var toolCalls: [ToolCall] = []

    init(modelID: String) {
        self.responseID = "resp_\(OpenAIID.random())"
        self.itemID = "msg_\(OpenAIID.random())"
        self.modelID = modelID
        self.created = OpenAIID.timestamp()
    }

    func encode(_ event: GenerationEvent) throws -> [SSEFrame] {
        switch event {
        case .started:
            return [frame("response.created", responseObject(status: "in_progress", includeOutput: false))]

        case .textDelta(let text):
            var frames: [SSEFrame] = []
            if !openedTextItem {
                openedTextItem = true
                frames.append(frame("response.output_item.added", .object([
                    ("type", .string("response.output_item.added")),
                    ("sequence_number", .int(nextSeq())),
                    ("output_index", .int(emittedItems)),
                    ("item", messageItem(status: "in_progress", withContent: false)),
                ])))
                frames.append(frame("response.content_part.added", .object([
                    ("type", .string("response.content_part.added")),
                    ("sequence_number", .int(nextSeq())),
                    ("item_id", .string(itemID)),
                    ("output_index", .int(emittedItems)),
                    ("content_index", .int(0)),
                    ("part", .object([("type", .string("output_text")), ("text", .string("")), ("annotations", .array([]))])),
                ])))
            }
            accumulatedText += text
            frames.append(frame("response.output_text.delta", .object([
                ("type", .string("response.output_text.delta")),
                ("sequence_number", .int(nextSeq())),
                ("item_id", .string(itemID)),
                ("output_index", .int(emittedItems)),
                ("content_index", .int(0)),
                ("delta", .string(text)),
            ])))
            return frames

        case .toolCall(let call):
            toolCalls.append(call)
            return [] // emitted at completion as function_call items

        case .completed(let result):
            finishReason = result.finishReason
            usage = result.usage
            return closeFrames()
        }
    }

    func terminator() -> [SSEFrame] { [] }  // Responses has no [DONE]; response.completed is terminal.

    // MARK: - Closing sequence

    private func closeFrames() -> [SSEFrame] {
        var frames: [SSEFrame] = []

        if openedTextItem {
            frames.append(frame("response.output_text.done", .object([
                ("type", .string("response.output_text.done")),
                ("sequence_number", .int(nextSeq())),
                ("item_id", .string(itemID)),
                ("output_index", .int(emittedItems)),
                ("content_index", .int(0)),
                ("text", .string(accumulatedText)),
            ])))
            frames.append(frame("response.content_part.done", .object([
                ("type", .string("response.content_part.done")),
                ("sequence_number", .int(nextSeq())),
                ("item_id", .string(itemID)),
                ("output_index", .int(emittedItems)),
                ("content_index", .int(0)),
                ("part", outputTextPart()),
            ])))
            frames.append(frame("response.output_item.done", .object([
                ("type", .string("response.output_item.done")),
                ("sequence_number", .int(nextSeq())),
                ("output_index", .int(emittedItems)),
                ("item", messageItem(status: "completed", withContent: true)),
            ])))
            emittedItems += 1
        }

        // Emit each tool call as a function_call output item.
        for call in toolCalls {
            frames.append(frame("response.output_item.added", .object([
                ("type", .string("response.output_item.added")),
                ("sequence_number", .int(nextSeq())),
                ("output_index", .int(emittedItems)),
                ("item", functionCallItem(call, status: "in_progress")),
            ])))
            frames.append(frame("response.function_call_arguments.delta", .object([
                ("type", .string("response.function_call_arguments.delta")),
                ("sequence_number", .int(nextSeq())),
                ("item_id", .string(call.id)),
                ("output_index", .int(emittedItems)),
                ("delta", .string(call.argumentsJSON)),
            ])))
            frames.append(frame("response.function_call_arguments.done", .object([
                ("type", .string("response.function_call_arguments.done")),
                ("sequence_number", .int(nextSeq())),
                ("item_id", .string(call.id)),
                ("output_index", .int(emittedItems)),
                ("arguments", .string(call.argumentsJSON)),
            ])))
            frames.append(frame("response.output_item.done", .object([
                ("type", .string("response.output_item.done")),
                ("sequence_number", .int(nextSeq())),
                ("output_index", .int(emittedItems)),
                ("item", functionCallItem(call, status: "completed")),
            ])))
            emittedItems += 1
        }

        frames.append(frame("response.completed", responseObject(status: "completed", includeOutput: true)))
        return frames
    }

    // MARK: - Object builders

    private func responseObject(status: String, includeOutput: Bool) -> OAIJSON {
        var responseFields: [(String, OAIJSON)] = [
            ("id", .string(responseID)),
            ("object", .string("response")),
            ("created_at", .int(created)),
            ("status", .string(status)),
            ("model", .string(modelID)),
        ]
        if includeOutput {
            responseFields.append(("output", .array(ResponsesPayload.outputItems(
                text: accumulatedText, hasText: openedTextItem, itemID: itemID, toolCalls: toolCalls))))
            responseFields.append(("usage", .object([
                ("input_tokens", .int(usage.promptTokens)),
                ("output_tokens", .int(usage.completionTokens)),
                ("total_tokens", .int(usage.totalTokens)),
            ])))
        } else {
            responseFields.append(("output", .array([])))
        }
        return .object([
            ("type", .string(status == "completed" ? "response.completed" : "response.created")),
            ("sequence_number", .int(nextSeq())),
            ("response", .object(responseFields)),
        ])
    }

    private func messageItem(status: String, withContent: Bool) -> OAIJSON {
        var fields: [(String, OAIJSON)] = [
            ("id", .string(itemID)),
            ("type", .string("message")),
            ("status", .string(status)),
            ("role", .string("assistant")),
        ]
        if withContent {
            fields.append(("content", .array([outputTextPart()])))
        }
        return .object(fields)
    }

    private func functionCallItem(_ call: ToolCall, status: String) -> OAIJSON {
        .object([
            ("id", .string(call.id)),
            ("type", .string("function_call")),
            ("status", .string(status)),
            ("name", .string(call.name)),
            ("arguments", .string(call.argumentsJSON)),
            ("call_id", .string(call.id)),
        ])
    }

    private func outputTextPart() -> OAIJSON {
        .object([
            ("type", .string("output_text")),
            ("text", .string(accumulatedText)),
            ("annotations", .array([])),
        ])
    }

    private func frame(_ event: String, _ data: OAIJSON) -> SSEFrame {
        SSEFrame(event: event, data: data.jsonString)
    }

    private func nextSeq() -> Int { defer { sequence += 1 }; return sequence }
}

/// Shared helpers for building the Responses `output` array (used by streaming + non-streaming).
enum ResponsesPayload {
    static func outputItems(text: String, hasText: Bool, itemID: String, toolCalls: [ToolCall]) -> [OAIJSON] {
        var items: [OAIJSON] = []
        if hasText {
            items.append(.object([
                ("type", .string("message")),
                ("id", .string(itemID)),
                ("status", .string("completed")),
                ("role", .string("assistant")),
                ("content", .array([.object([
                    ("type", .string("output_text")),
                    ("text", .string(text)),
                    ("annotations", .array([])),
                ])])),
            ]))
        }
        for call in toolCalls {
            items.append(.object([
                ("type", .string("function_call")),
                ("id", .string(call.id)),
                ("status", .string("completed")),
                ("name", .string(call.name)),
                ("arguments", .string(call.argumentsJSON)),
                ("call_id", .string(call.id)),
            ]))
        }
        return items
    }
}
