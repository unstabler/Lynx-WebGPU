import XCTest
import Metal
import QuartzCore
@testable import LynxWebGPUCore
@testable import LynxWebGPU

/// The **wiring** of in-flight frame accounting — whether commit and completion of a command buffer
/// carrying a drawable are reported to the coordinator.
///
/// The accounting rules themselves (the cap, no-negative, forgetting) are verified with no GPU by
/// `WGPUFrameCoordinatorTests`. What matters here is only whether the Metal path **is wired into them properly**.
final class SurfaceInFlightTests: XCTestCase {
    private var device: MTLDevice!

    override func setUpWithError() throws {
        device = MTLCreateSystemDefaultDevice()
        try XCTSkipIf(device == nil, "no Metal device")
    }

    // MARK: - Registration wiring

    func test_onlySwapchainSurfacesAreRegisteredForPacing() throws {
        let context = try LynxWebGPUContext(device: device)
        context.registerSurface(WGPUOffscreenSurface(
            identifier: "off", size: CGSize(width: 8, height: 8), device: device
        ))
        XCTAssertEqual(context.frameCoordinator.trackedCanvases, [], "an offscreen surface cannot fall behind")

        context.registerSurface(WGPUMetalLayerSurface(identifier: "screen", layer: CAMetalLayer()))
        XCTAssertEqual(context.frameCoordinator.trackedCanvases, ["screen"])
    }

    func test_theContextIsNotReadyWhenAnyCanvasIsSaturated() throws {
        let context = try LynxWebGPUContext(device: device)
        context.registerSurface(WGPUMetalLayerSurface(identifier: "sat", layer: CAMetalLayer()))
        XCTAssertTrue(context.isReadyForNextFrame)

        for _ in 0..<WGPUFrameCoordinator.defaultMaxFramesInFlight {
            context.frameCoordinator.noteCommitted(canvas: "sat")
        }
        XCTAssertFalse(context.isReadyForNextFrame)

        // Removing it leaves no dead counter behind — one left behind quietly stops the screen.
        context.unregisterSurface(identifier: "sat")
        XCTAssertTrue(context.isReadyForNextFrame)
    }

    // MARK: - Interpreter wiring

    /// Whether a batch carrying a drawable reports one commit and one completion — obtaining the texture
    /// several times from one surface is still one frame, so it must report once.
    func test_aBatchCarryingADrawableNotifiesCommitAndCompletionOnce() throws {
        let coordinator = CountingCoordinator()
        let context = try LynxWebGPUContext(device: device, frameCoordinator: coordinator)
        // Use an offscreen surface but register it for pacing directly — this guarantees a drawable even
        // headless while the notification path goes through the same code as a screen surface.
        context.registerSurface(WGPUOffscreenSurface(
            identifier: "count", size: CGSize(width: 8, height: 8), device: device
        ))
        coordinator.track(canvas: "count")

        let result = context.execute(commands: [
            ["op": "configureCanvas", "canvas": "count", "format": "rgba8unorm"],
            ["op": "getCurrentTexture", "id": 1, "canvas": "count"],
            ["op": "createTextureView", "id": 2, "texture": 1],
            ["op": "getCurrentTexture", "id": 3, "canvas": "count"],
            ["op": "beginRenderPass", "colorAttachments": [[
                "view": 2, "loadOp": "clear", "storeOp": "store",
                "clearValue": ["r": 0, "g": 0, "b": 0, "a": 1],
            ]]],
            ["op": "endPass"],
        ])
        XCTAssertEqual(result["ok"] as? Bool, true, "\(result)")

        XCTAssertEqual(coordinator.committedCount, 1, "frame accounting is once per canvas")
        XCTAssertTrue(waitUntil { coordinator.completedCount == 1 }, "the GPU completion must come back")
    }

    func test_aBatchWithNoDrawableNotifiesNothing() throws {
        let coordinator = CountingCoordinator()
        let context = try LynxWebGPUContext(device: device, frameCoordinator: coordinator)
        coordinator.track(canvas: "idle")

        let result = context.execute(commands: [
            ["op": "createBuffer", "id": 1, "size": 16, "usage": TestUsage.copyDst],
            ["op": "writeBuffer", "buffer": 1, "data": [Float](repeating: 0, count: 4).base64],
        ])
        XCTAssertEqual(result["ok"] as? Bool, true, "\(result)")

        XCTAssertTrue(waitUntil { context.stagingPool.pooledBufferCount == 1 }, "the batch really did run")
        XCTAssertEqual(coordinator.committedCount, 0)
        XCTAssertEqual(coordinator.completedCount, 0)
    }

    /// An internal **mid-frame** submit (`present: false`) is not a frame and is not counted —
    /// counting it would block the ticker on a single popErrorScope.
    func test_aPresentFalseBatchIsNotCountedInTheAccounting() throws {
        let coordinator = CountingCoordinator()
        let context = try LynxWebGPUContext(device: device, frameCoordinator: coordinator)
        context.registerSurface(WGPUOffscreenSurface(
            identifier: "mid", size: CGSize(width: 8, height: 8), device: device
        ))
        coordinator.track(canvas: "mid")

        let result = context.execute([
            "present": false,
            "commands": [
                ["op": "configureCanvas", "canvas": "mid", "format": "rgba8unorm"],
                ["op": "getCurrentTexture", "id": 1, "canvas": "mid"],
                ["op": "createTextureView", "id": 2, "texture": 1],
                ["op": "beginRenderPass", "colorAttachments": [[
                    "view": 2, "loadOp": "clear", "storeOp": "store",
                    "clearValue": ["r": 0, "g": 0, "b": 0, "a": 1],
                ]]],
                ["op": "endPass"],
            ],
        ])
        XCTAssertEqual(result["ok"] as? Bool, true, "\(result)")
        XCTAssertEqual(coordinator.committedCount, 0, "a mid-frame submit is not a frame")
    }

    // MARK: - CAMetalLayer round trip (only where a drawable comes out headless)

    func test_aCAMetalLayerSurfaceAlsoReturnsToZeroAfterAFrameRoundTrip() throws {
        let context = try LynxWebGPUContext(device: device)
        let surface = WGPUMetalLayerSurface(identifier: "layer", layer: CAMetalLayer())
        surface.updateDrawableSize(CGSize(width: 32, height: 32))
        context.registerSurface(surface)

        let result = context.execute(commands: [
            ["op": "configureCanvas", "canvas": "layer", "format": "bgra8unorm"],
            ["op": "getCurrentTexture", "id": 1, "canvas": "layer"],
            ["op": "createTextureView", "id": 2, "texture": 1],
            ["op": "beginRenderPass", "colorAttachments": [[
                "view": 2, "loadOp": "clear", "storeOp": "store",
                "clearValue": ["r": 0, "g": 0, "b": 0, "a": 1],
            ]]],
            ["op": "endPass"],
        ])
        try XCTSkipIf((result["ok"] as? Bool) != true, "could not obtain a drawable headless: \(result)")

        XCTAssertTrue(
            waitUntil { context.frameCoordinator.framesInFlight(canvas: "layer") == 0 },
            "the counter must come back after completion"
        )
        XCTAssertTrue(context.isReadyForNextFrame)
    }

    // MARK: - Canvas usage → backing texture

    /// A drawable is framebuffer-only **only** when the canvas is nothing but a render attachment.
    ///
    /// It used to key on `COPY_SRC` alone, so a canvas configured for `COPY_DST` (copying a render
    /// target in), `TEXTURE_BINDING` or `STORAGE_BINDING` (compute writing to the canvas) still got
    /// framebuffer-only drawables — Metal's validation layer asserts on the blit or shader access.
    func test_aLayerCanvasIsFramebufferOnlyOnlyForRenderAttachmentUsage() throws {
        let context = try LynxWebGPUContext(device: device)
        let surface = WGPUMetalLayerSurface(identifier: "usage", layer: CAMetalLayer())
        context.registerSurface(surface)

        let cases: [(usage: Int, framebufferOnly: Bool)] = [
            (TestUsage.renderAttachment, true),
            (TestUsage.renderAttachment | TestUsage.textureCopySrc, false),
            (TestUsage.renderAttachment | TestUsage.textureCopyDst, false),
            (TestUsage.renderAttachment | TestUsage.textureBinding, false),
            (TestUsage.renderAttachment | TestUsage.textureStorageBinding, false),
            (TestUsage.textureCopyDst, false),
            (TestUsage.textureStorageBinding, false),
        ]
        for (usage, expected) in cases {
            // Configure from the main thread (as the test runs) applies the layer properties synchronously.
            let result = context.execute(commands: [
                ["op": "configureCanvas", "canvas": "usage", "format": "bgra8unorm", "usage": usage],
            ])
            XCTAssertEqual(result["ok"] as? Bool, true, "\(result)")
            XCTAssertEqual(surface.layer.framebufferOnly, expected, "usage 0x\(String(usage, radix: 16))")
        }

        // Omitting usage means RENDER_ATTACHMENT (the spec default) — back to framebuffer-only.
        let result = context.execute(commands: [
            ["op": "configureCanvas", "canvas": "usage", "format": "bgra8unorm"],
        ])
        XCTAssertEqual(result["ok"] as? Bool, true, "\(result)")
        XCTAssertTrue(surface.layer.framebufferOnly)
    }

    /// `configure` from the JS thread lands `framebufferOnly` on main **later**, so the same batch can
    /// still get a framebuffer-only drawable. Rendering into it is fine; only a use Metal forbids on a
    /// framebuffer-only texture (here a copy out of it) fails — as a validation error rather than the
    /// Metal assertion — and once main has applied the configuration the same copy succeeds.
    func test_aCopyFromAStillFramebufferOnlyDrawableIsAValidationErrorNotACrash() throws {
        let context = try LynxWebGPUContext(device: device)
        let surface = WGPUMetalLayerSurface(identifier: "race", layer: CAMetalLayer())
        surface.updateDrawableSize(CGSize(width: 32, height: 32))
        context.registerSurface(surface)

        func frame(_ id: Int, copyInto buffer: Int? = nil) -> [[String: Any]] {
            var commands: [[String: Any]] = [
                ["op": "getCurrentTexture", "id": id, "canvas": "race"],
                ["op": "createTextureView", "id": id + 1, "texture": id],
                ["op": "beginRenderPass", "colorAttachments": [[
                    "view": id + 1, "loadOp": "clear", "storeOp": "store",
                    "clearValue": ["r": 0, "g": 0, "b": 0, "a": 1],
                ]]],
                ["op": "endPass"],
            ]
            if let buffer {
                commands.append(["op": "copyTextureToBuffer",
                                 "source": ["texture": id],
                                 "destination": ["buffer": buffer, "bytesPerRow": 256],
                                 "copySize": ["width": 1, "height": 1]])
            }
            return commands
        }
        func messages(_ result: [String: Any]) -> [String] {
            (result["errors"] as? [[String: Any]] ?? []).map { $0["message"] as? String ?? "" }
        }

        // On main: RENDER_ATTACHMENT only — device and format land synchronously.
        let warmup = context.execute(commands: [
            ["op": "configureCanvas", "canvas": "race", "format": "bgra8unorm"],
            ["op": "createBuffer", "id": 100, "size": 256, "usage": TestUsage.copyDst],
        ] + frame(1))
        try XCTSkipIf((warmup["ok"] as? Bool) != true, "could not obtain a drawable headless: \(warmup)")
        XCTAssertTrue(surface.layer.framebufferOnly)

        // From another thread while main is blocked: the layer update cannot land before the acquisition.
        func raced(_ commands: [[String: Any]]) -> [String: Any] {
            var result: [String: Any] = [:]
            let done = DispatchSemaphore(value: 0)
            DispatchQueue.global().async {
                result = context.execute(commands: commands)
                done.signal()
            }
            done.wait()
            return result
        }
        let configure: [String: Any] = [
            "op": "configureCanvas", "canvas": "race", "format": "bgra8unorm",
            "usage": TestUsage.renderAttachment | TestUsage.textureCopySrc,
        ]

        // Each frame waits for the previous ones to retire — a drained drawable pool would fail the
        // acquisition and read as a wrong result rather than an environment limit.
        func settle() {
            XCTAssertTrue(waitUntil { context.frameCoordinator.framesInFlight(canvas: "race") == 0 })
        }
        func skipIfNoDrawable(_ result: [String: Any]) throws {
            let lost = messages(result).contains { $0.contains("could not obtain a drawable") }
            try XCTSkipIf(lost, "could not obtain a drawable headless: \(result)")
        }
        settle()

        // Render-only use in that batch still works — a framebuffer-only drawable is a valid attachment.
        let renderOnly = raced([configure] + frame(10))
        try skipIfNoDrawable(renderOnly)
        XCTAssertTrue(surface.layer.framebufferOnly, "the layer update is still queued on main")
        XCTAssertEqual(renderOnly["ok"] as? Bool, true, "\(renderOnly)")
        settle()

        // A copy out of the canvas in such a batch fails alone, as validation.
        let withCopy = raced([configure] + frame(20, copyInto: 100))
        try skipIfNoDrawable(withCopy)
        XCTAssertTrue(surface.layer.framebufferOnly, "the layer update is still queued on main")
        XCTAssertEqual(withCopy["ok"] as? Bool, false, "\(withCopy)")
        XCTAssertEqual(messages(withCopy).count, 1, "only the copy fails: \(withCopy)")
        XCTAssertTrue(messages(withCopy).first?.contains("framebuffer-only") == true, "\(withCopy)")
        settle()

        // Once main has run, a later frame's drawable takes the same copy.
        RunLoop.main.run(until: Date().addingTimeInterval(0.05))
        XCTAssertFalse(surface.layer.framebufferOnly)
        var recovered = false
        for attempt in 0..<4 where !recovered {
            let result = context.execute(commands: frame(30 + attempt * 2, copyInto: 100))
            try skipIfNoDrawable(result)
            recovered = (result["ok"] as? Bool) == true
            settle()
        }
        XCTAssertTrue(recovered, "the copy must succeed once the configuration has landed")
    }

    /// The offscreen surface honours the canvas usage too — `STORAGE_BINDING` needs a writable texture.
    func test_anOffscreenCanvasWithStorageUsageIsShaderWritable() throws {
        let surface = WGPUOffscreenSurface(identifier: "store", size: CGSize(width: 8, height: 8), device: device)
        let context = try LynxWebGPUContext(device: device)
        context.registerSurface(surface)

        var result = context.execute(commands: [
            ["op": "configureCanvas", "canvas": "store", "format": "rgba8unorm"],
        ])
        XCTAssertEqual(result["ok"] as? Bool, true, "\(result)")
        XCTAssertFalse(try XCTUnwrap(surface.texture).usage.contains(.shaderWrite))

        result = context.execute(commands: [
            ["op": "configureCanvas", "canvas": "store", "format": "rgba8unorm",
             "usage": TestUsage.renderAttachment | TestUsage.textureStorageBinding],
        ])
        XCTAssertEqual(result["ok"] as? Bool, true, "\(result)")
        let texture = try XCTUnwrap(surface.texture)
        XCTAssertTrue(texture.usage.contains(.shaderWrite))
        XCTAssertTrue(texture.usage.contains(.renderTarget), "the harness still draws into it")

        // A resize rebuilds the texture with the same usage.
        surface.updateDrawableSize(CGSize(width: 16, height: 16))
        XCTAssertTrue(try XCTUnwrap(surface.texture).usage.contains(.shaderWrite))
    }

    /// **The process must not die** even when the size arrives as NaN, infinity or negative.
    ///
    /// The size comes from UI layout (`bounds × pixelRatio`) — a frame before measurement or a strange
    /// pixelRatio leaks a NaN at some point. Passed straight down, the offscreen surface's
    /// `Int(size.width)` kills the process with **a Swift runtime trap**. It is instant death rather
    /// than a validation error, so not even a log survives.
    func test_aStrangeResizeIsIgnoredWithoutCrashing() throws {
        let context = try LynxWebGPUContext(device: device)
        try context.attachOffscreenCanvas(identifier: "odd", size: CGSize(width: 8, height: 8))

        for bad in [
            CGSize(width: CGFloat.nan, height: -5),
            CGSize(width: 8, height: CGFloat.nan),
            CGSize(width: CGFloat.infinity, height: 8),
            CGSize(width: -16, height: -16),
        ] {
            context.resizeCanvas(identifier: "odd", drawableSize: bad)
        }

        // The surface is uncorrupted and still at its original size.
        let info = context.canvasInfo(identifier: "odd")
        XCTAssertEqual(info["ok"] as? Bool, true, "\(info)")
        XCTAssertEqual(info["width"] as? Int, 8)
        XCTAssertEqual(info["height"] as? Int, 8)

        // A normal size still applies — ignoring did not lock the surface.
        context.resizeCanvas(identifier: "odd", drawableSize: CGSize(width: 16, height: 16))
        XCTAssertEqual(context.canvasInfo(identifier: "odd")["width"] as? Int, 16)
    }

    /// A strange size at attach time is **rejected as a validation error** (here there is a channel to return one).
    func test_attachingAnOffscreenCanvasWithAStrangeSizeIsRejected() throws {
        let context = try LynxWebGPUContext(device: device)
        XCTAssertThrowsError(
            try context.attachOffscreenCanvas(
                identifier: "nan", size: CGSize(width: CGFloat.nan, height: 8)
            )
        )
    }

    private func waitUntil(timeout: TimeInterval = 2, _ condition: () -> Bool) -> Bool {
        let deadline = Date().addingTimeInterval(timeout)
        while Date() < deadline {
            if condition() { return true }
            Thread.sleep(forTimeInterval: 0.005)
        }
        return condition()
    }
}

/// A coordinator recording notification counts — it only checks whether the interpreter **calls** the accounting.
private final class CountingCoordinator: WGPUFrameCoordinator {
    private let counterLock = NSLock()
    private var committed = 0
    private var completed = 0

    override func noteCommitted(canvas: String) {
        counterLock.lock()
        committed += 1
        counterLock.unlock()
        super.noteCommitted(canvas: canvas)
    }

    override func noteCompleted(canvas: String) {
        counterLock.lock()
        completed += 1
        counterLock.unlock()
        super.noteCompleted(canvas: canvas)
    }

    var committedCount: Int {
        counterLock.lock()
        defer { counterLock.unlock() }
        return committed
    }

    var completedCount: Int {
        counterLock.lock()
        defer { counterLock.unlock() }
        return completed
    }
}
