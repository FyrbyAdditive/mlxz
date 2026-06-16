import Testing
import Foundation
@testable import MLXZCore

@Suite struct ImageContentTests {
    @Test func httpURLBecomesImageURL() {
        let part = ImageContent.part(fromURLString: "https://example.com/cat.png")
        guard case let .imageURL(url) = part else {
            Issue.record("expected .imageURL"); return
        }
        #expect(url.absoluteString == "https://example.com/cat.png")
    }

    @Test func base64DataURLBecomesImageData() {
        // "hi" base64-encoded.
        let b64 = Data("hi".utf8).base64EncodedString()
        let part = ImageContent.part(fromURLString: "data:image/png;base64,\(b64)")
        guard case let .imageData(data) = part else {
            Issue.record("expected .imageData"); return
        }
        #expect(String(data: data, encoding: .utf8) == "hi")
    }

    @Test func malformedDataURLReturnsNil() {
        #expect(ImageContent.part(fromURLString: "data:image/png;base64,!!!notbase64!!!") == nil)
        #expect(ImageContent.part(fromURLString: "data:nocomma") == nil)
    }
}
