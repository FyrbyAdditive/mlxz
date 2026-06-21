import Foundation
import Testing

@testable import MLXZInference

/// `MLXModelDownloader` retries transient network failures so one timed-out shard doesn't abort a
/// 30GB+ multi-shard snapshot (the gemma-4-31b-8bit download that "always failed" at a shard boundary).
@Suite struct DownloadRetryTests {
    private func urlError(_ code: Int) -> NSError {
        NSError(domain: NSURLErrorDomain, code: code)
    }

    @Test func timeoutIsTransient() {
        #expect(MLXModelDownloader.isTransientNetworkError(urlError(NSURLErrorTimedOut)))
    }

    @Test func connectionLostAndOfflineAreTransient() {
        #expect(MLXModelDownloader.isTransientNetworkError(urlError(NSURLErrorNetworkConnectionLost)))
        #expect(MLXModelDownloader.isTransientNetworkError(urlError(NSURLErrorNotConnectedToInternet)))
        #expect(MLXModelDownloader.isTransientNetworkError(urlError(NSURLErrorCannotConnectToHost)))
        #expect(MLXModelDownloader.isTransientNetworkError(urlError(NSURLErrorDNSLookupFailed)))
    }

    @Test func permanentErrorsAreNotRetried() {
        // 404 / bad-URL / auth-type failures shouldn't be retried (retrying won't fix them).
        #expect(!MLXModelDownloader.isTransientNetworkError(urlError(NSURLErrorBadURL)))
        #expect(!MLXModelDownloader.isTransientNetworkError(urlError(NSURLErrorUnsupportedURL)))
        #expect(!MLXModelDownloader.isTransientNetworkError(
            NSError(domain: "SomeOtherDomain", code: 404)))
    }

    @Test func makeHubClientSucceeds() {
        // The tuned client is constructed without throwing (smoke check).
        _ = MLXModelDownloader.makeHubClient()
    }
}
