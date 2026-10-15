import XCTest
import Metal
import ImageIO
import CoreGraphics
import UniformTypeIdentifiers
import LynxWebGPUCore
@testable import LynxWebGPU

/// External image → texture (`createImageBitmap` + `copyExternalImageToTexture`).
///
/// On the web the browser owns decoding. Here ImageIO does, so **channel order and vertical
/// orientation** can go quietly wrong — both only surface on screen, so they are pinned by pixels.
final class ExternalImageTests: XCTestCase {
    var harness: RenderHarness!

    override func setUpWithError() throws {
        try XCTSkipIf(MTLCreateSystemDefaultDevice() == nil, "no Metal device")
        harness = try XCTUnwrap(RenderHarness.make())
    }

    override func tearDown() {
        harness = nil
        super.tearDown()
    }

    // MARK: - Building the image

    /// Builds a PNG whose top and bottom halves differ — it must be asymmetric to see flipY actually flip.
    private func makePNG(
        width: Int, height: Int,
        top: (UInt8, UInt8, UInt8, UInt8), bottom: (UInt8, UInt8, UInt8, UInt8)
    ) throws -> Data {
        var pixels = [UInt8]()
        for row in 0..<height {
            let color = row < height / 2 ? top : bottom
            for _ in 0..<width { pixels.append(contentsOf: [color.0, color.1, color.2, color.3]) }
        }
        let provider = CGDataProvider(data: Data(pixels) as CFData)!
        let image = CGImage(
            width: width, height: height, bitsPerComponent: 8, bitsPerPixel: 32,
            bytesPerRow: width * 4, space: CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.last.rawValue
                                     | CGBitmapInfo.byteOrder32Big.rawValue),
            provider: provider, decode: nil, shouldInterpolate: false, intent: .defaultIntent
        )!
        let output = NSMutableData()
        let destination = CGImageDestinationCreateWithData(
            output, UTType.png.identifier as CFString, 1, nil
        )!
        CGImageDestinationAddImage(destination, image, nil)
        XCTAssertTrue(CGImageDestinationFinalize(destination))
        return output as Data
    }

    /// Waits for `decodeImage` synchronously (test convenience).
    @discardableResult
    private func decode(
        handle: Int, data: Data, options: WGPUImageDecoder.Options = .init()
    ) throws -> [String: Any] {
        let expectation = expectation(description: "decodeImage")
        var payload: [String: Any] = [:]
        harness.runtime.decodeImage(
            handle: handle, data: data, name: nil, options: options, provider: nil
        ) { result in
            payload = result
            expectation.fulfill()
        }
        wait(for: [expectation], timeout: 5)
        return payload
    }

    // MARK: - Decoding

    func test_decodingAPNGGivesTheRightSizeAndChannelOrder() throws {
        let png = try makePNG(width: 4, height: 4, top: (255, 0, 0, 255), bottom: (0, 0, 255, 255))
        let bitmap = try WGPUImageDecoder.decode(png)

        XCTAssertEqual(bitmap.width, 4)
        XCTAssertEqual(bitmap.height, 4)
        XCTAssertEqual(bitmap.bytesPerRow, 16)
        // The first row is red — with the wrong RGBA order B would come out 255 here.
        XCTAssertEqual(Array(bitmap.pixels.prefix(4)), [255, 0, 0, 255], "the first pixel is not red")
        // The last row is blue — a vertical flip is caught here.
        XCTAssertEqual(Array(bitmap.pixels.suffix(4)), [0, 0, 255, 255], "it came out flipped vertically")
    }

    func test_flipYFlipsTopToBottom() throws {
        let png = try makePNG(width: 4, height: 4, top: (255, 0, 0, 255), bottom: (0, 0, 255, 255))
        let flipped = try WGPUImageDecoder.decode(png, options: .init(flipY: true))

        XCTAssertEqual(Array(flipped.pixels.prefix(4)), [0, 0, 255, 255], "flipped, the first row is blue")
        XCTAssertEqual(Array(flipped.pixels.suffix(4)), [255, 0, 0, 255])
    }

    func test_premultiplyAlphaMultipliesAlphaIntoTheColor() throws {
        // White at 50% alpha. Premultiplied it lands near 128; unpremultiplied it stays 255.
        let png = try makePNG(width: 2, height: 2, top: (255, 255, 255, 128), bottom: (255, 255, 255, 128))

        let straight = try WGPUImageDecoder.decode(png)
        XCTAssertEqual(straight.pixels[0], 255, "unpremultiplied leaves the color as it is")

        let premultiplied = try WGPUImageDecoder.decode(png, options: .init(premultiplyAlpha: true))
        XCTAssertEqual(Int(premultiplied.pixels[0]), 128, accuracy: 2)
        XCTAssertTrue(premultiplied.premultiplied)
    }

    func test_resizeChangesTheSize() throws {
        let png = try makePNG(width: 8, height: 8, top: (255, 0, 0, 255), bottom: (255, 0, 0, 255))
        let bitmap = try WGPUImageDecoder.decode(png, options: .init(resize: (width: 4, height: 2)))

        XCTAssertEqual(bitmap.width, 4)
        XCTAssertEqual(bitmap.height, 2)
        XCTAssertEqual(bitmap.pixels.count, 4 * 2 * 4)
    }

    /// Broken data comes back **as an error** — silently producing an empty image blackens the screen and loses the cause.
    func test_undecodableDataIsAnError() {
        XCTAssertThrowsError(try WGPUImageDecoder.decode(Data([0x00, 0x01, 0x02, 0x03]))) { error in
            XCTAssertTrue("\(error)".contains("decode"), "\(error)")
        }
        XCTAssertThrowsError(try WGPUImageDecoder.decode(Data()))
    }

    func test_decodeImageRegistersUnderTheHandleAndReturnsTheSize() throws {
        let png = try makePNG(width: 8, height: 4, top: (0, 255, 0, 255), bottom: (0, 255, 0, 255))
        let result = try decode(handle: 40, data: png)

        XCTAssertEqual(result["ok"] as? Bool, true, "\(result)")
        XCTAssertEqual(result["width"] as? Int, 8)
        XCTAssertEqual(result["height"] as? Int, 4)
    }

    func test_rejectsWithNeitherImageBytesNorAName() throws {
        let expectation = expectation(description: "decodeImage")
        var payload: [String: Any] = [:]
        harness.runtime.decodeImage(
            handle: 40, data: nil, name: nil, options: .init(), provider: nil
        ) { payload = $0; expectation.fulfill() }
        wait(for: [expectation], timeout: 5)

        XCTAssertEqual(payload["ok"] as? Bool, false)
    }

    // MARK: - Uploading into a texture

    private static let shader = """
    @group(0) @binding(0) var tex: texture_2d<f32>;
    @group(0) @binding(1) var samp: sampler;

    struct Out {
        @builtin(position) position: vec4f,
        @location(0) uv: vec2f,
    };

    @vertex
    fn vs_main(@builtin(vertex_index) index: u32) -> Out {
        var positions = array<vec2f, 3>(vec2f(-1.0, -1.0), vec2f(3.0, -1.0), vec2f(-1.0, 3.0));
        var out: Out;
        out.position = vec4f(positions[index], 0.0, 1.0);
        // Texture coordinates have 0 at the top — flip y so the image's first row lands at the top of the screen.
        out.uv = vec2f(positions[index].x * 0.5 + 0.5, 0.5 - positions[index].y * 0.5);
        return out;
    }

    @fragment
    fn fs_main(in: Out) -> @location(0) vec4f {
        return textureSample(tex, samp, in.uv);
    }
    """

    /// The command stream uploading an image into a texture and spreading it across the screen.
    private func drawImage(bitmap: Int, texture size: (Int, Int), copy: [String: Any]) {
        var commands: [[String: Any]] = [
            ["op": "configureCanvas", "canvas": "test", "format": "rgba8unorm"],
            ["op": "createShaderModule", "id": 1, "code": Self.shader],
            ["op": "createTexture", "id": 2, "size": ["width": size.0, "height": size.1],
             "format": "rgba8unorm",
             "usage": TestUsage.textureBinding | TestUsage.textureCopyDst],
        ]
        commands.append(copy)
        commands.append(contentsOf: [
            ["op": "createTextureView", "id": 3, "texture": 2],
            ["op": "createSampler", "id": 4],
            ["op": "createRenderPipeline", "id": 5, "layout": "auto",
             "vertex": ["module": 1, "entryPoint": "vs_main"],
             "fragment": ["module": 1, "entryPoint": "fs_main", "targets": [["format": "rgba8unorm"]]]],
            ["op": "getBindGroupLayout", "id": 6, "pipeline": 5, "index": 0],
            ["op": "createBindGroup", "id": 7, "layout": 6, "entries": [
                ["binding": 0, "resource": ["textureView": 3]],
                ["binding": 1, "resource": ["sampler": 4]],
            ]],
            ["op": "getCurrentTexture", "id": 10, "canvas": "test"],
            ["op": "createTextureView", "id": 11, "texture": 10],
            ["op": "beginRenderPass", "colorAttachments": [[
                "view": 11, "loadOp": "clear", "storeOp": "store",
                "clearValue": ["r": 0, "g": 1, "b": 0, "a": 1],
            ]]],
            ["op": "setPipeline", "pipeline": 5],
            ["op": "setBindGroup", "index": 0, "bindGroup": 7],
            ["op": "draw", "vertexCount": 3],
            ["op": "endPass"],
        ])
        _ = bitmap
        harness.executeExpectingSuccess(commands)
    }

    func test_uploadsADecodedImageIntoATextureAndSamplesIt() throws {
        let png = try makePNG(width: 4, height: 4, top: (255, 0, 0, 255), bottom: (0, 0, 255, 255))
        try decode(handle: 40, data: png)

        drawImage(bitmap: 40, texture: (4, 4), copy: [
            "op": "copyExternalImageToTexture",
            "source": ["source": 40],
            "destination": ["texture": 2],
            "copySize": ["width": 4, "height": 4],
        ])

        // The top of the screen is the image's first row (red), the bottom its last (blue).
        try harness.assertPixel(x: 32, y: 8, equals: (255, 0, 0, 255), "the top half")
        try harness.assertPixel(x: 32, y: 56, equals: (0, 0, 255, 255), "the bottom half")
    }

    /// Uploads only a crop of the source — using the full image width as the stride shifts the color here.
    func test_uploadsOnlyACroppedPartOfTheImage() throws {
        let png = try makePNG(width: 8, height: 8, top: (255, 0, 0, 255), bottom: (0, 0, 255, 255))
        try decode(handle: 40, data: png)

        drawImage(bitmap: 40, texture: (4, 4), copy: [
            "op": "copyExternalImageToTexture",
            "source": ["source": 40, "origin": ["x": 2, "y": 4]],   // 4x4 out of the bottom half (blue)
            "destination": ["texture": 2],
            "copySize": ["width": 4, "height": 4],
        ])

        try harness.assertPixel(x: 32, y: 32, equals: (0, 0, 255, 255), "the cropped region is entirely blue")
    }

    /// Spec `GPUCopyExternalImageSourceInfo.flipY` — flipping at **copy time**.
    ///
    /// A different place from `createImageBitmap`'s flipY. Web libraries use this one
    /// (three.js's `Texture.flipY` defaults to true, so ignoring it flips textures silently).
    func test_copyTimeFlipYFlipsTopToBottom() throws {
        let png = try makePNG(width: 4, height: 4, top: (255, 0, 0, 255), bottom: (0, 0, 255, 255))
        try decode(handle: 40, data: png)

        drawImage(bitmap: 40, texture: (4, 4), copy: [
            "op": "copyExternalImageToTexture",
            "source": ["source": 40, "flipY": true],
            "destination": ["texture": 2],
            "copySize": ["width": 4, "height": 4],
        ])

        // Flipped, the top of the screen is the image's **last** row (blue).
        try harness.assertPixel(x: 32, y: 8, equals: (0, 0, 255, 255), "the top half")
        try harness.assertPixel(x: 32, y: 56, equals: (255, 0, 0, 255), "the bottom half")
    }

    /// Flipping at **both** decode time and copy time returns to the original — evidence that the two
    /// options live in different layers.
    func test_flippingAtBothDecodeAndCopyReturnsToTheOriginal() throws {
        let png = try makePNG(width: 4, height: 4, top: (255, 0, 0, 255), bottom: (0, 0, 255, 255))
        try decode(handle: 40, data: png, options: .init(flipY: true))

        drawImage(bitmap: 40, texture: (4, 4), copy: [
            "op": "copyExternalImageToTexture",
            "source": ["source": 40, "flipY": true],
            "destination": ["texture": 2],
        ])

        try harness.assertPixel(x: 32, y: 8, equals: (255, 0, 0, 255), "the original, unchanged")
        try harness.assertPixel(x: 32, y: 56, equals: (0, 0, 255, 255))
    }

    /// With a partial copy and flipY together, the flip happens **only within the cropped region**.
    func test_flipYAppliesWithinTheRegionForAPartialCopy() throws {
        // Top 2 rows red, bottom 2 blue. Taking 4x2 from (0,1) gives 1 red row + 1 blue row.
        let png = try makePNG(width: 4, height: 4, top: (255, 0, 0, 255), bottom: (0, 0, 255, 255))
        try decode(handle: 40, data: png)

        drawImage(bitmap: 40, texture: (4, 2), copy: [
            "op": "copyExternalImageToTexture",
            "source": ["source": 40, "origin": ["x": 0, "y": 1], "flipY": true],
            "destination": ["texture": 2],
            "copySize": ["width": 4, "height": 2],
        ])

        // Source row 1 (red) and row 2 (blue) → flipped, blue comes first.
        try harness.assertPixel(x: 32, y: 8, equals: (0, 0, 255, 255), "top")
        try harness.assertPixel(x: 32, y: 56, equals: (255, 0, 0, 255), "bottom")
    }

    func test_omittingCopySizeMeansTheWholeImage() throws {
        let png = try makePNG(width: 4, height: 4, top: (255, 0, 0, 255), bottom: (255, 0, 0, 255))
        try decode(handle: 40, data: png)

        drawImage(bitmap: 40, texture: (4, 4), copy: [
            "op": "copyExternalImageToTexture",
            "source": ["source": 40],
            "destination": ["texture": 2],
        ])

        try harness.assertPixel(x: 32, y: 32, equals: (255, 0, 0, 255))
    }

    func test_rejectsACopyPastTheImage() throws {
        let png = try makePNG(width: 4, height: 4, top: (255, 0, 0, 255), bottom: (255, 0, 0, 255))
        try decode(handle: 40, data: png)

        let result = harness.execute([
            ["op": "createTexture", "id": 2, "size": ["width": 8, "height": 8],
             "format": "rgba8unorm", "usage": TestUsage.textureBinding | TestUsage.textureCopyDst],
            ["op": "copyExternalImageToTexture",
             "source": ["source": 40],
             "destination": ["texture": 2],
             "copySize": ["width": 8, "height": 8]],
        ])
        XCTAssertEqual(result["ok"] as? Bool, false)
        XCTAssertTrue(harness.describeErrors(result).contains("exceeds the image"), harness.describeErrors(result))
    }

    /// Runs one `copyExternalImageToTexture` into a fresh 4x4 texture and expects a validation error
    /// containing `needle` — never a crash, never a pixel read outside the decoded image.
    private func assertCopyRejected(
        source: [String: Any], destination: [String: Any] = [:], copySize: [String: Any]?,
        containing needle: String, file: StaticString = #filePath, line: UInt = #line
    ) {
        var copy: [String: Any] = [
            "op": "copyExternalImageToTexture",
            "source": source.merging(["source": 40]) { $1 },
            "destination": destination.merging(["texture": 2]) { $1 },
        ]
        if let copySize { copy["copySize"] = copySize }
        let result = harness.execute([
            ["op": "createTexture", "id": 2, "size": ["width": 4, "height": 4],
             "format": "rgba8unorm", "usage": TestUsage.textureBinding | TestUsage.textureCopyDst],
            copy,
            ["op": "destroy", "id": 2],
        ])
        let errors = harness.describeErrors(result)
        XCTAssertEqual(result["ok"] as? Bool, false, "\(copy)", file: file, line: line)
        XCTAssertTrue(errors.contains("[validation]"), errors, file: file, line: line)
        XCTAssertTrue(errors.contains(needle), errors, file: file, line: line)
    }

    /// A negative source origin used to slip past the `origin + size <= image` check and `memcpy`
    /// from **before** the decoded pixels — an out-of-bounds heap read uploaded into the texture.
    func test_rejectsANegativeSourceOrigin() throws {
        let png = try makePNG(width: 4, height: 4, top: (255, 0, 0, 255), bottom: (0, 0, 255, 255))
        try decode(handle: 40, data: png)

        let size: [String: Any] = ["width": 2, "height": 2]
        assertCopyRejected(source: ["origin": ["x": -1, "y": 0]], copySize: size, containing: "must not be negative")
        assertCopyRejected(source: ["origin": ["x": 0, "y": -2]], copySize: size, containing: "must not be negative")
        assertCopyRejected(source: ["origin": ["x": -4, "y": -4]], copySize: nil, containing: "must not be negative")
    }

    /// Values that would overflow `origin + size` or `rowBytes * height` are a validation error, not a trap.
    func test_hugeOrNegativeCopyValuesAreAValidationErrorNotATrap() throws {
        let png = try makePNG(width: 4, height: 4, top: (255, 0, 0, 255), bottom: (0, 0, 255, 255))
        try decode(handle: 40, data: png)

        assertCopyRejected(source: ["origin": ["x": 1, "y": 0]], copySize: ["width": Int.max, "height": 1],
                           containing: "exceeds the image")
        assertCopyRejected(source: ["origin": ["x": Int.max, "y": Int.max]], copySize: ["width": 1, "height": 1],
                           containing: "exceeds the image")
        assertCopyRejected(source: ["origin": ["x": Int.max, "y": 0]], copySize: nil, containing: "exceeds the image")
        assertCopyRejected(source: [:], copySize: ["width": 2, "height": Int.max], containing: "exceeds the image")
        assertCopyRejected(source: [:], copySize: ["width": -1, "height": 2], containing: "must not be negative")
        assertCopyRejected(source: [:], copySize: ["width": 2, "height": 2, "depthOrArrayLayers": 2],
                           containing: "single layer")
    }

    /// The destination end is checked before it reaches the unsigned Metal blit arguments.
    func test_rejectsADestinationOutsideTheTexture() throws {
        let png = try makePNG(width: 4, height: 4, top: (255, 0, 0, 255), bottom: (0, 0, 255, 255))
        try decode(handle: 40, data: png)

        let size: [String: Any] = ["width": 2, "height": 2]
        assertCopyRejected(source: [:], destination: ["origin": ["x": -1, "y": 0]], copySize: size,
                           containing: "destination origin must not be negative")
        assertCopyRejected(source: [:], destination: ["origin": ["x": 3, "y": 0]], copySize: size,
                           containing: "exceeds the destination texture")
        assertCopyRejected(source: [:], destination: ["origin": ["x": 0, "y": Int.max]], copySize: size,
                           containing: "exceeds the destination texture")
        assertCopyRejected(source: [:], destination: ["mipLevel": 1], copySize: size, containing: "mipLevel")
    }

    /// It cannot upload into a compressed texture — the GPU has no block encoder.
    func test_cannotUploadIntoACompressedTexture() throws {
        try XCTSkipUnless(
            WGPUDeviceCapability.supportsCompression(.astc4x4Unorm, on: harness.context!.device),
            "this device does not support ASTC"
        )
        let png = try makePNG(width: 4, height: 4, top: (255, 0, 0, 255), bottom: (255, 0, 0, 255))
        try decode(handle: 40, data: png)

        let result = harness.execute([
            ["op": "createTexture", "id": 2, "size": ["width": 4, "height": 4],
             "format": "astc-4x4-unorm", "usage": TestUsage.textureBinding | TestUsage.textureCopyDst],
            ["op": "copyExternalImageToTexture",
             "source": ["source": 40], "destination": ["texture": 2]],
        ])
        XCTAssertEqual(result["ok"] as? Bool, false)
        XCTAssertTrue(harness.describeErrors(result).contains("compressed"), harness.describeErrors(result))
    }

    /// A non-4-byte format is refused rather than going quietly wrong.
    func test_a16BitFormatIsRefused() throws {
        let png = try makePNG(width: 4, height: 4, top: (255, 0, 0, 255), bottom: (255, 0, 0, 255))
        try decode(handle: 40, data: png)

        let result = harness.execute([
            ["op": "createTexture", "id": 2, "size": ["width": 4, "height": 4],
             "format": "rgba16float", "usage": TestUsage.textureBinding | TestUsage.textureCopyDst],
            ["op": "copyExternalImageToTexture",
             "source": ["source": 40], "destination": ["texture": 2]],
        ])
        XCTAssertEqual(result["ok"] as? Bool, false)
        XCTAssertTrue(harness.describeErrors(result).contains("4-byte"), harness.describeErrors(result))
    }

    func test_aMissingImageHandleIsAClearError() throws {
        let result = harness.execute([
            ["op": "createTexture", "id": 2, "size": ["width": 4, "height": 4],
             "format": "rgba8unorm", "usage": TestUsage.textureBinding | TestUsage.textureCopyDst],
            ["op": "copyExternalImageToTexture",
             "source": ["source": 99], "destination": ["texture": 2]],
        ])
        XCTAssertEqual(result["ok"] as? Bool, false)
        XCTAssertTrue(harness.describeErrors(result).contains("ImageBitmap"), harness.describeErrors(result))
    }

    /// Whether the `destroy` op really releases the native pixels (JS `bitmap.close()`).
    func test_destroyReleasesTheImage() throws {
        let png = try makePNG(width: 4, height: 4, top: (255, 0, 0, 255), bottom: (255, 0, 0, 255))
        try decode(handle: 40, data: png)

        let result = harness.execute([
            ["op": "destroy", "id": 40],
            ["op": "createTexture", "id": 2, "size": ["width": 4, "height": 4],
             "format": "rgba8unorm", "usage": TestUsage.textureBinding | TestUsage.textureCopyDst],
            ["op": "copyExternalImageToTexture",
             "source": ["source": 40], "destination": ["texture": 2]],
        ])
        XCTAssertEqual(result["ok"] as? Bool, false)
    }
}
