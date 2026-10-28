import XCTest
import Metal
import LynxWebGPUCore
@testable import LynxWebGPU

/// `WGPUSharedTextureCache` on its own — one texture per (device, key), coalesced requests, failures
/// that reach every waiter, and entries that live exactly as long as someone holds them.
///
/// `make` here is a stand-in that allocates a small texture, so these tests are about the cache's
/// bookkeeping; decoding and uploading are covered in `SharedTextureTests`.
final class SharedTextureCacheTests: XCTestCase {
    var device: MTLDevice!
    var cache: WGPUSharedTextureCache!

    override func setUpWithError() throws {
        device = try XCTUnwrap(MTLCreateSystemDefaultDevice(), "no Metal device")
        cache = WGPUSharedTextureCache()
    }

    private func makeTexture(on device: MTLDevice) throws -> MTLTexture {
        let descriptor = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .rgba8Unorm, width: 4, height: 4, mipmapped: false
        )
        return try XCTUnwrap(device.makeTexture(descriptor: descriptor))
    }

    /// Acquires and waits for the result.
    private func acquire(
        _ request: WGPUSharedTextureRequest, on device: MTLDevice? = nil,
        make: (() throws -> MTLTexture)? = nil
    ) -> Result<WGPUSharedTexture, WGPUError> {
        let target = device ?? self.device!
        let done = expectation(description: "acquire")
        var result: Result<WGPUSharedTexture, WGPUError>!
        cache.acquire(request, device: target, make: make ?? { try self.makeTexture(on: target) }) {
            result = $0
            done.fulfill()
        }
        wait(for: [done], timeout: 5)
        return result
    }

    private let request = WGPUSharedTextureRequest(key: "clouds", name: "clouds.png")

    func test_aSecondRequestGetsTheSameTextureWithoutMakingAnother() throws {
        let first = try acquire(request).get()
        let second = try acquire(request).get()
        XCTAssertTrue(first === second)
        XCTAssertTrue(first.texture === second.texture, "one MTLTexture for the key")
        XCTAssertEqual(cache.makeCount, 1)
        XCTAssertEqual(cache.liveEntryCount, 1)
    }

    /// Requests that arrive while the first one is still decoding wait for it — no second decode.
    func test_concurrentRequestsShareOneMake() throws {
        let gate = DispatchSemaphore(value: 0)
        let both = expectation(description: "both waiters")
        both.expectedFulfillmentCount = 2
        var results: [WGPUSharedTexture] = []
        let resultsLock = NSLock()
        let make = { () throws -> MTLTexture in
            gate.wait()
            return try self.makeTexture(on: self.device)
        }
        for _ in 0..<2 {
            cache.acquire(request, device: device, make: make) { result in
                resultsLock.lock()
                if case .success(let entry) = result { results.append(entry) }
                resultsLock.unlock()
                both.fulfill()
            }
        }
        XCTAssertEqual(cache.makeCount, 1, "the second request joined the first")
        gate.signal()
        wait(for: [both], timeout: 5)

        XCTAssertEqual(results.count, 2)
        XCTAssertTrue(results[0] === results[1])
    }

    /// A failed make reaches every waiter with the same error and leaves nothing behind.
    func test_aFailureReachesEveryWaiterAndTheNextRequestTriesAgain() throws {
        let gate = DispatchSemaphore(value: 0)
        let both = expectation(description: "both waiters")
        both.expectedFulfillmentCount = 2
        var messages: [String] = []
        let messagesLock = NSLock()
        let failing = { () throws -> MTLTexture in
            gate.wait()
            throw WGPUError.validation("could not decode the image")
        }
        for _ in 0..<2 {
            cache.acquire(request, device: device, make: failing) { result in
                messagesLock.lock()
                if case .failure(let error) = result { messages.append(error.message) }
                messagesLock.unlock()
                both.fulfill()
            }
        }
        gate.signal()
        wait(for: [both], timeout: 5)

        XCTAssertEqual(messages, ["could not decode the image", "could not decode the image"])
        XCTAssertEqual(cache.liveEntryCount, 0)

        // The file may have arrived since — a later request decodes again.
        _ = try acquire(request).get()
        XCTAssertEqual(cache.makeCount, 2)
    }

    /// The cache holds entries weakly — when the last owner lets go, the entry is gone.
    func test_theEntryGoesAwayWithItsLastOwner() throws {
        var first: WGPUSharedTexture? = try acquire(request).get()
        var second: WGPUSharedTexture? = try acquire(request).get()
        XCTAssertEqual(cache.liveEntryCount, 1)

        first = nil
        XCTAssertEqual(cache.liveEntryCount, 1, "one owner still holds it")
        XCTAssertNotNil(cache.entry(forKey: "clouds", device: device))

        second = nil
        XCTAssertEqual(cache.liveEntryCount, 0)
        XCTAssertNil(cache.entry(forKey: "clouds", device: device))
        _ = (first, second)

        _ = try acquire(request).get()
        XCTAssertEqual(cache.makeCount, 2, "the next request makes it again")
    }

    /// Contexts on different devices never share — a texture belongs to its device.
    func test_differentDevicesGetSeparateEntries() throws {
        let devices = MTLCopyAllDevices()
        try XCTSkipIf(devices.count < 2, "only one Metal device on this machine")
        let a = try acquire(request, on: devices[0]).get()
        let b = try acquire(request, on: devices[1]).get()
        XCTAssertFalse(a === b)
        XCTAssertEqual(cache.makeCount, 2)
        XCTAssertEqual(cache.liveEntryCount, 2)
    }

    /// The same key asking for different pixels breaks the caller's contract: the first texture wins,
    /// and the cache says so.
    func test_aKeyReusedForOtherPixelsWarnsAndKeepsTheFirst() throws {
        var warnings: [String] = []
        cache.warnConflict = { warnings.append($0) }

        let first = try acquire(WGPUSharedTextureRequest(key: "bg", name: "sunset.png")).get()
        _ = try acquire(WGPUSharedTextureRequest(key: "bg", name: "sunset.png", label: "another label")).get()
        XCTAssertEqual(warnings, [], "a different label is the same pixels")

        let other = try acquire(WGPUSharedTextureRequest(key: "bg", name: "night.png", resizeWidth: 64)).get()
        XCTAssertTrue(other === first, "the first content wins")
        XCTAssertEqual(other.request.name, "sunset.png")
        XCTAssertEqual(warnings.count, 1)
        XCTAssertTrue(warnings[0].contains("'sunset.png'") && warnings[0].contains("'night.png' at 64x"), warnings[0])
        XCTAssertEqual(cache.makeCount, 1)
    }
}
