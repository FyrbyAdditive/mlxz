// The fixed prompt suite for losslessness gating and content-type benchmarks.
// Checked in (not generated) so results are comparable across runs and machines.
// Suites: chat (open-ended), code, math — acceptance rates differ a lot by content type,
// so speedups are reported per suite. The long-context case is synthesized by the bench
// (filler + question) to avoid a multi-MB file in the repo.

enum BenchPrompts {
    static let chat: [String] = [
        "What are the main tradeoffs between renting and buying a home? Keep it under 300 words.",
        "Summarize the plot of Romeo and Juliet in one paragraph, then list three themes.",
        "Explain to a 10-year-old why the sky is blue and sunsets are red.",
        "Schreibe eine kurze E-Mail an einen Kollegen, in der du ein Meeting auf Donnerstag verschiebst.",
        "Write a six-line poem about a lighthouse keeper who is afraid of the dark.",
        "I want to get into running but I hate mornings. Give me a realistic weekly plan.",
    ]

    static let code: [String] = [
        "Write a Python function that merges two sorted lists into one sorted list without using sort().",
        "Refactor this into a list comprehension:\n\nresult = []\nfor x in items:\n    if x % 2 == 0:\n        result.append(x * x)\n",
        "Write a SQL query that finds the three highest-spending customers per region from tables orders(customer_id, region, amount) and customers(id, name).",
        "Write a regex that matches ISO-8601 dates (YYYY-MM-DD) but rejects month 00 or 13+ and day 00 or 32+. Explain each part.",
        "This Swift code crashes with 'Index out of range'. Find the bug:\n\nvar names = [\"a\", \"b\", \"c\"]\nfor i in 0...names.count {\n    print(names[i])\n}\n",
        "Convert this JSON to a YAML equivalent and explain any type ambiguities:\n\n{\"name\": \"test\", \"count\": 3, \"tags\": [\"a\", \"b\"], \"nested\": {\"on\": true}}",
    ]

    static let math: [String] = [
        "A train leaves at 9:15 and travels 240 km at 80 km/h. A second train leaves the same station at 10:00 at 120 km/h on the same route. When does the second train catch up?",
        "Solve for x: 3(x - 4) + 7 = 2x + 11. Show each step.",
        "A recipe for 6 people needs 450 g flour. How much flour for 10 people? Convert the answer to ounces (1 oz = 28.35 g).",
        "What is the probability of rolling at least one six in four rolls of a fair die? Show the reasoning.",
        "Sketch a proof that the square root of 2 is irrational.",
        "A shop discounts a jacket by 20%, then adds 10% tax. Is the final price the same as taxing first and then discounting? Prove it in general.",
    ]

    static let all: [(suite: String, prompts: [String])] = [
        ("chat", chat), ("code", code), ("math", math),
    ]
}
