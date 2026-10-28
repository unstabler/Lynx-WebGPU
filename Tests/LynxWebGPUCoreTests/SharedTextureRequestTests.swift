import XCTest
import CoreGraphics
import QuartzCore
@testable import LynxWebGPUCore

/// `importSharedTexture`'s Core half — the payload the bridge receives, the resize rule, and the
/// "exactly one callback" guarantee. The bridge is iOS-only, so this is where its parsing is checked.
final class SharedTextureRequestTests: XCTestCase {

    // MARK: - Parsing

    func test_readsTheJSPayload() throws {
        let request = try WGPUSharedTextureRequest(from: WGPUValueReader([
            "key": "clouds@2x", "url": "file:///tmp/clouds.png",
            "resizeWidth": 640, "resizeHeight": 480, "label": "clouds",
        ]))
        XCTAssertEqual(request, WGPUSharedTextureRequest(
            key: "clouds@2x", name: "file:///tmp/clouds.png", resizeWidth: 640, resizeHeight: 480, label: "clouds"
        ))
    }

    func test_optionalFieldsMayBeLeftOut() throws {
        let request = try WGPUSharedTextureRequest(from: WGPUValueReader(["key": "k", "url": "a.png"]))
        XCTAssertNil(request.resizeWidth)
        XCTAssertNil(request.resizeHeight)
        XCTAssertNil(request.label)
    }

    func test_rejectsAMissingOrEmptyKeyOrURL() {
        let payloads: [[String: Any]] = [
            ["url": "a.png"],
            ["key": "", "url": "a.png"],
            ["key": "k"],
            ["key": "k", "url": ""],
        ]
        for payload in payloads {
            XCTAssertThrowsError(try WGPUSharedTextureRequest(from: WGPUValueReader(payload)), "\(payload)") { error in
                XCTAssertEqual((error as? WGPUError)?.kind, .validation, "\(payload)")
            }
        }
    }

    func test_rejectsAResizeBelowOne() {
        for field in ["resizeWidth", "resizeHeight"] {
            for value in [0, -3] {
                XCTAssertThrowsError(try WGPUSharedTextureRequest(from: WGPUValueReader([
                    "key": "k", "url": "a.png", field: value,
                ]))) { error in
                    XCTAssertEqual((error as? WGPUError)?.kind, .validation)
                    XCTAssertEqual((error as? WGPUError)?.path, field)
                }
            }
        }
    }

    // MARK: - The resize rule

    func test_bothSizesGivenAreTakenAsTheyAre() {
        let request = WGPUSharedTextureRequest(key: "k", name: "a", resizeWidth: 100, resizeHeight: 300)
        let size = request.targetSize(sourceWidth: 3370, sourceHeight: 3917)
        XCTAssertEqual(size.width, 100)
        XCTAssertEqual(size.height, 300, "keeping the aspect ratio is the caller's call")
    }

    func test_oneSizeGivenDerivesTheOtherFromTheAspectRatioRoundedUp() {
        let byWidth = WGPUSharedTextureRequest(key: "k", name: "a", resizeWidth: 1000)
            .targetSize(sourceWidth: 3370, sourceHeight: 3917)
        XCTAssertEqual(byWidth.width, 1000)
        XCTAssertEqual(byWidth.height, 1163, "3917 × 1000 / 3370 = 1162.3, rounded up as HTML createImageBitmap does")

        let byHeight = WGPUSharedTextureRequest(key: "k", name: "a", resizeHeight: 1000)
            .targetSize(sourceWidth: 3370, sourceHeight: 3917)
        XCTAssertEqual(byHeight.width, 861, "3370 × 1000 / 3917 = 860.4, rounded up")
        XCTAssertEqual(byHeight.height, 1000)

        let sliver = WGPUSharedTextureRequest(key: "k", name: "a", resizeWidth: 1)
            .targetSize(sourceWidth: 4000, sourceHeight: 10)
        XCTAssertEqual(sliver.height, 1, "a derived axis never reaches 0")

        let exact = WGPUSharedTextureRequest(key: "k", name: "a", resizeWidth: 10)
            .targetSize(sourceWidth: 40, sourceHeight: 20)
        XCTAssertEqual(exact.height, 5, "an exact quotient is not bumped up")
    }

    /// A resize near the 2^53 the payload reader accepts, on an aspect ratio past 1024:1, derives an
    /// axis beyond `Int` — it saturates rather than trapping, and the runtime's size limit rejects it.
    func test_aDerivedAxisPastIntSaturatesInsteadOfTrapping() {
        let largest = (1 << 53) - 1
        let tall = WGPUSharedTextureRequest(key: "k", name: "a", resizeWidth: largest)
            .targetSize(sourceWidth: 1, sourceHeight: 2000)
        XCTAssertEqual(tall.width, largest)
        XCTAssertEqual(tall.height, Int.max, "2000 × (2^53 − 1) is past Int.max")

        let wide = WGPUSharedTextureRequest(key: "k", name: "a", resizeHeight: largest)
            .targetSize(sourceWidth: 2000, sourceHeight: 1)
        XCTAssertEqual(wide.width, Int.max)
        XCTAssertEqual(wide.height, largest)
    }

    func test_noSizeGivenKeepsTheSourceSize() {
        let size = WGPUSharedTextureRequest(key: "k", name: "a").targetSize(sourceWidth: 1807, sourceHeight: 3917)
        XCTAssertEqual(size.width, 1807)
        XCTAssertEqual(size.height, 3917)
    }

    func test_samePixelsMeansTheSameSourceAndRequestedSize() {
        let base = WGPUSharedTextureRequest(key: "k", name: "a.png", resizeWidth: 10, label: "one")
        XCTAssertTrue(base.describesSamePixels(as: WGPUSharedTextureRequest(key: "other", name: "a.png", resizeWidth: 10, label: "two")),
                      "the key and the label do not change the pixels")
        XCTAssertFalse(base.describesSamePixels(as: WGPUSharedTextureRequest(key: "k", name: "b.png", resizeWidth: 10)))
        XCTAssertFalse(base.describesSamePixels(as: WGPUSharedTextureRequest(key: "k", name: "a.png", resizeWidth: 20)))
        XCTAssertFalse(base.describesSamePixels(as: WGPUSharedTextureRequest(key: "k", name: "a.png")))
    }

    // MARK: - The bridge entry point

    /// Runs `start` and returns every callback it made — the contract is exactly one.
    private func callbacks(_ params: [String: Any], runtime: WebGPURuntime) -> [[String: Any]] {
        var results: [[String: Any]] = []
        WGPUSharedTextureImport.start(params, runtime: runtime, provider: nil) { results.append($0) }
        return results
    }

    func test_aBrokenPayloadCallsBackOnceWithAValidationError() {
        let runtime = NoSharingRuntime()
        for params in [["key": "k", "url": "a.png"], ["id": 3, "url": "a.png"]] as [[String: Any]] {
            let results = callbacks(params, runtime: runtime)
            XCTAssertEqual(results.count, 1, "\(params)")
            XCTAssertEqual(results.first?["ok"] as? Bool, false)
            let kind = ((results.first?["errors"] as? [[String: Any]])?.first?["kind"]) as? String
            XCTAssertEqual(kind, "validation", "\(params)")
        }
    }

    /// A runtime that does not implement shared textures still answers — `unsupported`, once — which
    /// is also what shows `start` parsed the request and handed it on.
    func test_aRuntimeWithoutSharedTexturesAnswersUnsupported() {
        let results = callbacks(["id": 3, "key": "k", "url": "a.png"], runtime: NoSharingRuntime())
        XCTAssertEqual(results.count, 1)
        XCTAssertEqual(results.first?["ok"] as? Bool, false)
        let kind = ((results.first?["errors"] as? [[String: Any]])?.first?["kind"]) as? String
        XCTAssertEqual(kind, "unsupported")
    }
}

/// The smallest `WebGPURuntime` — it does not implement `importSharedTexture`, so the protocol's
/// default answers for it.
private final class NoSharingRuntime: WebGPURuntime {
    func execute(_ payload: [String: Any]) -> [String: Any] { ["ok": true] }
    func adapterInfo() -> [String: Any] { [:] }
    func shaderCompilationInfo(handle: Int) -> [String: Any] { [:] }
    func canvasInfo(identifier: String) -> [String: Any] { [:] }
    func readBuffer(handle: Int, offset: Int, size: Int?, completion: @escaping ([String: Any]) -> Void) {}
    func decodeImage(
        handle: Int, data: Data?, name: String?, options: WGPUImageDecodeOptions,
        provider: WGPUAssetProvider?, completion: @escaping ([String: Any]) -> Void
    ) {}
    func attachCanvas(identifier: String, layer: CAMetalLayer) {}
    func attachOffscreenCanvas(identifier: String, size: CGSize) throws {}
    func resizeCanvas(identifier: String, drawableSize: CGSize) {}
    func detachCanvas(identifier: String) {}
    func readCanvasPixels(identifier: String) throws -> WGPUPixelReadback {
        throw WGPUError.unsupported("no canvases")
    }
    var isReadyForNextFrame: Bool { true }
    func reset() {}
}
