import Testing

@testable import MLXZCore

/// Contract for ThinkParser: separate `<think>…</think>` reasoning from the visible answer in a
/// streamed token feed. Reasoning leaking into chat content (with a stray `</think>`) is the bug
/// this prevents. The hard requirements:
///   - explicit `<think>…</think>` blocks are routed to `reasoning`, never `visibleText`;
///   - a generation that begins inside thinking (leading `</think>` with no open tag) is handled;
///   - a PLAIN reply (no think tags) streams as visible immediately — never buffered waiting for a
///     `</think>` that never comes;
///   - tags split across chunk boundaries are handled;
///   - concatenating the visible deltas == the answer; concatenating reasoning deltas == the CoT.
@Suite struct ThinkParserTests {

    /// Feed `chunks` through a fresh parser, return concatenated (visible, reasoning).
    private func run(_ chunks: [String], startInsideThink: Bool = false)
        -> (visible: String, reasoning: String)
    {
        var p = ThinkParser(startInsideThink: startInsideThink)
        var vis = "", rea = ""
        for c in chunks {
            let o = p.consume(c)
            vis += o.visibleText
            rea += o.reasoning
        }
        let tail = p.finish()
        vis += tail.visibleText
        rea += tail.reasoning
        return (vis, rea)
    }

    @Test func explicitBlockWholeInOneChunk() {
        let (v, r) = run(["<think>reasoning here</think>The answer."])
        #expect(v == "The answer.")
        #expect(r == "reasoning here")
    }

    @Test func preOpenedReasoningNoOpenTag() {
        // Generation began inside <think> (template pre-opened it): stream starts with reasoning,
        // terminated by </think>.
        let (v, r) = run(["The user said hi. I'll greet.</think>Hi! How can I help?"])
        #expect(v == "Hi! How can I help?")
        #expect(r == "The user said hi. I'll greet.")
    }

    @Test func userLeakedCaseReproduction() {
        // The exact shape the user saw leak: reasoning, a newline, </think>, then the answer.
        let (v, r) = run([
            "The user is just saying \"test\". I'll acknowledge it briefly.\n</think>\n\nHi! I'm here and ready to help."
        ])
        #expect(v == "\n\nHi! I'm here and ready to help.")
        #expect(r == "The user is just saying \"test\". I'll acknowledge it briefly.\n")
    }

    @Test func plainReplyStreamsAsVisible() {
        // No think tags at all → everything is visible, nothing held back as reasoning.
        let (v, r) = run(["Hello", " there", ", friend!"])
        #expect(v == "Hello there, friend!")
        #expect(r == "")
    }

    @Test func plainReplyIsNotFullyBuffered() {
        // A plain reply must emit incrementally, not wait for end-of-stream. After feeding a chunk
        // with no tag, most of it should already be visible (only a short partial-tag tail held).
        var p = ThinkParser()
        let o = p.consume("This is a normal answer with no thinking whatsoever.")
        #expect(o.reasoning.isEmpty)
        #expect(o.visibleText.count >= "This is a normal answer with no thinking whatsoever.".count - "</think>".count)
    }

    @Test func openTagSplitAcrossChunks() {
        let (v, r) = run(["<th", "ink>cot", "</think>answer"])
        #expect(v == "answer")
        #expect(r == "cot")
    }

    @Test func closeTagSplitAcrossChunks() {
        let (v, r) = run(["<think>cot part", "</th", "ink>visible"])
        #expect(v == "visible")
        #expect(r == "cot part")
    }

    @Test func reasoningStreamedAcrossChunks() {
        let (v, r) = run(["<think>step one ", "step two ", "step three</think>", "done"])
        #expect(v == "done")
        #expect(r == "step one step two step three")
    }

    @Test func leadingWhitespaceBeforeOpenTag() {
        // Text before an explicit <think> is preserved verbatim as visible (we never silently eat
        // content); here that's the leading whitespace. The real-model shape puts whitespace AFTER
        // </think> (part of the answer), covered by userLeakedCaseReproduction.
        let (v, r) = run(["\n  <think>cot</think>answer"])
        #expect(v == "\n  answer")
        #expect(r == "cot")
    }

    @Test func textBeforeInlineThinkIsVisible() {
        // Not pre-opened: real text, then a think block, then more text. The leading text is visible.
        let (v, r) = run(["Sure. <think>let me reason</think> Here you go."])
        #expect(v == "Sure.  Here you go.")
        #expect(r == "let me reason")
    }

    @Test func unterminatedThinkAtEnd() {
        // `<think>` opened but stream ended before `</think>`: it was all reasoning.
        let (v, r) = run(["<think>incomplete reasoning"])
        #expect(v == "")
        #expect(r == "incomplete reasoning")
    }

    @Test func emptyThinkBlock() {
        let (v, r) = run(["<think></think>answer"])
        #expect(v == "answer")
        #expect(r == "")
    }

    @Test func angleBracketTextThatIsNotAThinkTag() {
        // A `<` that isn't a think tag must not be swallowed.
        let (v, r) = run(["a < b and c > d, also 1<2"])
        #expect(v == "a < b and c > d, also 1<2")
        #expect(r == "")
    }

    // MARK: - Pre-opened think block (template pre-opens `<think>`, prose streams across chunks)

    @Test func preOpenedReasoningStreamedAcrossChunks() {
        // The real-model shape: the template pre-opened `<think>` (no open tag in the stream), the
        // model writes reasoning as plain prose across MANY chunks, and `</think>` arrives only at
        // the end. With startInsideThink the prose is all reasoning, the post-tag text is the answer.
        let (v, r) = run(
            ["Here's a thinking process:\n\n", "1. Understand the problem. ", "2. Compute 17*23. ",
             "3. The answer is 391.", "\n</think>\n\n", "17 * 23 = 391."],
            startInsideThink: true)
        #expect(r == "Here's a thinking process:\n\n1. Understand the problem. 2. Compute 17*23. 3. The answer is 391.\n")
        #expect(v == "\n\n17 * 23 = 391.")
    }

    @Test func preOpenedReasoningRegressionWithVisibleStart() {
        // WITHOUT startInsideThink (the old behavior), the same streamed prose leaks into visible
        // because each chunk is flushed before the late `</think>` arrives — this is the quirk fixed
        // by starting inside the block. Documents why the flag is necessary.
        let (v, _) = run(
            ["Here's a thinking process:\n\n", "1. Understand. ", "2. Compute.",
             "\n</think>\n\n", "Answer: 391."],
            startInsideThink: false)
        #expect(v.contains("Here's a thinking process"))  // leaked — the bug startInsideThink fixes
    }

    @Test func preOpenedThenForcedCloseTransition() {
        // The exact stream our reasoning-budget force-close produces: pre-opened reasoning, then our
        // injected transition string + `</think>`, then the answer. All reasoning before `</think>`,
        // answer after.
        let (v, r) = run(
            ["Step one. ", "Step two. ",
             "\nConsidering the limited time, I'll answer based on the above.\n</think>\n\n",
             "The final answer is 42."],
            startInsideThink: true)
        #expect(v == "\n\nThe final answer is 42.")
        #expect(r == "Step one. Step two. \nConsidering the limited time, I'll answer based on the above.\n")
    }

    @Test func preOpenedUnterminatedIsAllReasoning() {
        // Pre-opened think that never closes (model stopped mid-reasoning): everything is reasoning,
        // nothing leaks to visible.
        let (v, r) = run(["thinking ", "and more thinking"], startInsideThink: true)
        #expect(v == "")
        #expect(r == "thinking and more thinking")
    }
}
