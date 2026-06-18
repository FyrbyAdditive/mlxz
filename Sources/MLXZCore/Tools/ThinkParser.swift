import Foundation

/// A streaming splitter that separates a reasoning model's `<think>…</think>` chain-of-thought
/// from the visible answer. Qwen3.5/3.6 reasoning models (when the client's prompt asks them to,
/// e.g. VS Code Copilot) wrap reasoning in `<think>` tags; without this the reasoning — and a stray
/// `</think>` — leaks into the chat content.
///
/// Contract:
///   - text inside `<think>…</think>` is routed to `reasoning`, never to `visibleText`;
///   - a PLAIN reply (no think tags) streams as `visibleText` immediately — it is never buffered
///     waiting for a `</think>` that will never arrive (only a short partial-tag tail is held);
///   - a bare leading `</think>` (the model was prompted to think and the opening tag was consumed
///     by the template) is treated defensively: everything before it is reasoning — a lone
///     `</think>` is never legitimate visible content;
///   - tags split across chunk boundaries are handled.
///
/// IMPORTANT — pre-opened think blocks: Qwen3.5/3.6's chat template PRE-OPENS `<think>\n` on the
/// assistant turn, so the opening tag is in the *prompt*, never in the generated stream — the model
/// starts already inside the reasoning block and only ever emits the *closing* `</think>`. Worse,
/// some checkpoints (e.g. Qwen3.6-27B-4bit) write the whole reasoning as plain prose and emit
/// `</think>` only late (or not at all). If the parser started in `.visible`, that prose would be
/// flushed as visible content chunk-by-chunk *before* the late `</think>` ever arrived — so the
/// "bare leading `</think>`" rule can't reclaim it. Callers that know thinking is on (the template
/// pre-opened the block) MUST construct the parser with `startInsideThink: true` so all prose is
/// routed to reasoning until `</think>` — independent of chunk timing.
///
/// Feed deltas via `consume(_:)`; call `finish()` at end-of-stream to flush.
public struct ThinkParser: Sendable {
    public struct Output: Sendable, Equatable {
        public var visibleText: String
        public var reasoning: String
        public init(visibleText: String = "", reasoning: String = "") {
            self.visibleText = visibleText
            self.reasoning = reasoning
        }
    }

    private static let openTag = "<think>"
    private static let closeTag = "</think>"

    private enum Mode {
        case visible        // outside a think block (the default; plain replies live here)
        case insideThink    // between <think> and </think>
    }

    private var mode: Mode
    private var buffer = ""

    /// - Parameter startInsideThink: begin in the reasoning block (the chat template pre-opened
    ///   `<think>` so the stream starts inside it and only ever emits the closing tag). Default false
    ///   (plain replies / no pre-opened think).
    public init(startInsideThink: Bool = false) {
        self.mode = startInsideThink ? .insideThink : .visible
    }

    public mutating func consume(_ chunk: String) -> Output {
        buffer += chunk
        var out = Output()
        process(&out, atEnd: false)
        return out
    }

    /// Flush at end of stream. Remaining buffered text is emitted per the current mode.
    public mutating func finish() -> Output {
        var out = Output()
        process(&out, atEnd: true)
        if !buffer.isEmpty {
            switch mode {
            case .visible: out.visibleText += buffer       // plain reply / trailing answer
            case .insideThink: out.reasoning += buffer      // unterminated reasoning
            }
            buffer = ""
        }
        return out
    }

    private mutating func process(_ out: inout Output, atEnd: Bool) {
        var progress = true
        while progress {
            progress = false
            switch mode {
            case .visible:
                // Whichever think marker comes first, in buffer order. A bare `</think>` ahead of
                // any `<think>` means reasoning was pre-opened (template consumed the open tag), so
                // everything before it is reasoning.
                let open = buffer.range(of: Self.openTag)
                let close = buffer.range(of: Self.closeTag)
                if close != nil, (open == nil || close!.lowerBound < open!.lowerBound) {
                    out.reasoning += String(buffer[buffer.startIndex ..< close!.lowerBound])
                    buffer.removeSubrange(buffer.startIndex ..< close!.upperBound)
                    // We were implicitly inside a think block; now back to visible.
                    progress = true
                } else if let open {
                    out.visibleText += String(buffer[buffer.startIndex ..< open.lowerBound])
                    buffer.removeSubrange(buffer.startIndex ..< open.upperBound)
                    mode = .insideThink
                    progress = true
                } else {
                    // No complete marker: emit visible, holding only a tail that could begin one.
                    let keep = partialTagSuffix(buffer, tags: [Self.openTag, Self.closeTag], atEnd: atEnd)
                    if keep < buffer.count {
                        let idx = buffer.index(buffer.startIndex, offsetBy: buffer.count - keep)
                        out.visibleText += String(buffer[buffer.startIndex ..< idx])
                        buffer.removeSubrange(buffer.startIndex ..< idx)
                    }
                }
            case .insideThink:
                if let close = buffer.range(of: Self.closeTag) {
                    out.reasoning += String(buffer[buffer.startIndex ..< close.lowerBound])
                    buffer.removeSubrange(buffer.startIndex ..< close.upperBound)
                    mode = .visible
                    progress = true
                } else {
                    let keep = partialTagSuffix(buffer, tags: [Self.closeTag], atEnd: atEnd)
                    if keep < buffer.count {
                        let idx = buffer.index(buffer.startIndex, offsetBy: buffer.count - keep)
                        out.reasoning += String(buffer[buffer.startIndex ..< idx])
                        buffer.removeSubrange(buffer.startIndex ..< idx)
                    }
                }
            }
        }
    }

    /// Length of the trailing suffix of `text` to retain because it could be the start of one of
    /// `tags` arriving in a later chunk. Zero at end-of-stream (nothing more is coming).
    private func partialTagSuffix(_ text: String, tags: [String], atEnd: Bool) -> Int {
        if atEnd { return 0 }
        let chars = Array(text)
        var keep = 0
        for tag in tags {
            let tagChars = Array(tag)
            let maxKeep = min(chars.count, tagChars.count - 1)
            for len in stride(from: maxKeep, through: 1, by: -1) {
                if Array(chars[(chars.count - len)...]) == Array(tagChars[0 ..< len]) {
                    keep = max(keep, len)
                    break
                }
            }
        }
        return keep
    }
}
