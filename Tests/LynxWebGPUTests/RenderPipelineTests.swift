import XCTest
import Metal
import LynxWebGPUCore
@testable import LynxWebGPU

/// Runs the whole WGSL → MSL → Metal pipeline on the GPU and verifies it **by pixel**.
final class RenderPipelineTests: XCTestCase {
    private var harness: RenderHarness!

    override func setUpWithError() throws {
        try XCTSkipIf(MTLCreateSystemDefaultDevice() == nil, "no Metal device")
        harness = try XCTUnwrap(RenderHarness.make())
    }

    override func tearDown() {
        harness = nil
        super.tearDown()
    }

    // MARK: - Triangle

    private static let triangleShader = """
    struct VertexOutput {
        @builtin(position) position: vec4f,
        @location(0) color: vec3f,
    };

    @vertex
    fn vs_main(@location(0) position: vec2f, @location(1) color: vec3f) -> VertexOutput {
        var out: VertexOutput;
        out.position = vec4f(position, 0.0, 1.0);
        out.color = color;
        return out;
    }

    @fragment
    fn fs_main(in: VertexOutput) -> @location(0) vec4f {
        return vec4f(in.color, 1.0);
    }
    """

    func test_theTriangleIsDrawnAndDistinctFromTheClearColor() throws {
        // Interleaved position (x, y) + color (r, g, b) — stride 20B
        let vertices: [Float] = [
            -0.5, -0.5, 1, 0, 0,
             0.5, -0.5, 1, 0, 0,
             0.0,  0.5, 1, 0, 0,
        ]

        harness.executeExpectingSuccess([
            ["op": "configureCanvas", "canvas": "test", "format": "rgba8unorm"],
            ["op": "createShaderModule", "id": 1, "code": Self.triangleShader],
            ["op": "createBuffer", "id": 2, "usage": TestUsage.vertex | TestUsage.copyDst,
             "data": vertices.base64],
            ["op": "createRenderPipeline", "id": 3, "layout": "auto",
             "vertex": [
                "module": 1, "entryPoint": "vs_main",
                "buffers": [[
                    "arrayStride": 20,
                    "attributes": [
                        ["format": "float32x2", "offset": 0, "shaderLocation": 0],
                        ["format": "float32x3", "offset": 8, "shaderLocation": 1],
                    ],
                ]],
             ],
             "fragment": [
                "module": 1, "entryPoint": "fs_main",
                "targets": [["format": "rgba8unorm"]],
             ]],
            ["op": "getCurrentTexture", "id": 10, "canvas": "test"],
            ["op": "createTextureView", "id": 11, "texture": 10],
            ["op": "beginRenderPass", "colorAttachments": [[
                "view": 11, "loadOp": "clear", "storeOp": "store",
                "clearValue": ["r": 0.0, "g": 0.0, "b": 1.0, "a": 1.0],
            ]]],
            ["op": "setPipeline", "pipeline": 3],
            ["op": "setVertexBuffer", "slot": 0, "buffer": 2],
            ["op": "draw", "vertexCount": 3],
            ["op": "endPass"],
        ])

        // The center is inside the triangle → red. The top-left corner is outside → the clear color (blue).
        try harness.assertPixel(x: 32, y: 32, equals: (255, 0, 0, 255), "inside the triangle")
        try harness.assertPixel(x: 1, y: 1, equals: (0, 0, 255, 255), "outside the triangle (the clear color)")
    }

    func test_aUniformBufferReachesTheFragmentOutput() throws {
        let shader = """
        struct Tint { color: vec4f };
        @group(0) @binding(0) var<uniform> tint: Tint;

        @vertex
        fn vs_main(@builtin(vertex_index) index: u32) -> @builtin(position) vec4f {
            // A triangle covering the whole screen (no vertex buffer, driven by vertex_index alone)
            var positions = array<vec2f, 3>(
                vec2f(-1.0, -1.0),
                vec2f( 3.0, -1.0),
                vec2f(-1.0,  3.0),
            );
            return vec4f(positions[index], 0.0, 1.0);
        }

        @fragment
        fn fs_main() -> @location(0) vec4f {
            return tint.color;
        }
        """
        let tint: [Float] = [0.0, 1.0, 0.0, 1.0]   // green

        harness.executeExpectingSuccess([
            ["op": "configureCanvas", "canvas": "test", "format": "rgba8unorm"],
            ["op": "createShaderModule", "id": 1, "code": shader],
            ["op": "createBuffer", "id": 2, "size": 16,
             "usage": TestUsage.uniform | TestUsage.copyDst, "data": tint.base64],
            ["op": "createRenderPipeline", "id": 3, "layout": "auto",
             "vertex": ["module": 1, "entryPoint": "vs_main"],
             "fragment": ["module": 1, "entryPoint": "fs_main", "targets": [["format": "rgba8unorm"]]]],
            ["op": "getBindGroupLayout", "id": 4, "pipeline": 3, "index": 0],
            ["op": "createBindGroup", "id": 5, "layout": 4,
             "entries": [["binding": 0, "resource": ["buffer": 2]]]],
            ["op": "getCurrentTexture", "id": 10, "canvas": "test"],
            ["op": "createTextureView", "id": 11, "texture": 10],
            ["op": "beginRenderPass", "colorAttachments": [[
                "view": 11, "loadOp": "clear", "storeOp": "store",
                "clearValue": ["r": 0.0, "g": 0.0, "b": 0.0, "a": 1.0],
            ]]],
            ["op": "setPipeline", "pipeline": 3],
            ["op": "setBindGroup", "index": 0, "bindGroup": 5],
            ["op": "draw", "vertexCount": 3],
            ["op": "endPass"],
        ])

        try harness.assertPixel(x: 32, y: 32, equals: (0, 255, 0, 255), "the uniform color")
        try harness.assertPixel(x: 5, y: 60, equals: (0, 255, 0, 255), "it must cover the whole screen")
    }

    /// Checks the layout of an integer `vec3` uniform **by the value the GPU read**.
    ///
    /// WGSL `vec3<i32>` is 12 bytes; MSL `int3` is 16. Without the emitter using `packed_int3`, the
    /// following fields shift by 4 bytes and **a different value is read with no error** — transpiler
    /// tests only reach "the string matches and it compiles", so whether it really points at the same
    /// place is checked here.
    ///
    /// (`packed_int3`/`packed_uint3` exist in MSL and are 12 bytes. If the name matched but the size
    /// were 16, this test would break.)
    func test_anIntegerVec3UniformReadsAtTheWGSLOffsets() throws {
        let shader = """
        struct Counts {
            offsets: vec3<i32>,   // offset 0  (12B)
            total: i32,           // offset 12
            sizes: vec3<u32>,     // offset 16 (12B)
            stride: u32,          // offset 28
        };
        @group(0) @binding(0) var<uniform> counts: Counts;

        @vertex
        fn vs_main(@builtin(vertex_index) index: u32) -> @builtin(position) vec4f {
            var positions = array<vec2f, 3>(vec2f(-1.0, -1.0), vec2f(3.0, -1.0), vec2f(-1.0, 3.0));
            return vec4f(positions[index], 0.0, 1.0);
        }

        @fragment
        fn fs_main() -> @location(0) vec4f {
            let a = counts.offsets.x + counts.offsets.y + counts.offsets.z + counts.total;
            let b = counts.sizes.x + counts.sizes.y + counts.sizes.z + counts.stride;
            let c = counts.offsets.z * 10 + i32(counts.sizes.y);
            return vec4f(f32(a) / 255.0, f32(b) / 255.0, f32(c) / 255.0, 1.0);
        }
        """
        // offsets(1,2,3) total 4 · sizes(5,6,7) stride 8 — filled at the WGSL offsets exactly.
        let values: [Int32] = [1, 2, 3, 4, 5, 6, 7, 8]
        let data = values.withUnsafeBufferPointer { Data(buffer: $0).base64EncodedString() }

        harness.executeExpectingSuccess([
            ["op": "configureCanvas", "canvas": "test", "format": "rgba8unorm"],
            ["op": "createShaderModule", "id": 1, "code": shader],
            ["op": "createBuffer", "id": 2, "size": 32,
             "usage": TestUsage.uniform | TestUsage.copyDst, "data": data],
            ["op": "createRenderPipeline", "id": 3, "layout": "auto",
             "vertex": ["module": 1, "entryPoint": "vs_main"],
             "fragment": ["module": 1, "entryPoint": "fs_main", "targets": [["format": "rgba8unorm"]]]],
            ["op": "getBindGroupLayout", "id": 4, "pipeline": 3, "index": 0],
            ["op": "createBindGroup", "id": 5, "layout": 4,
             "entries": [["binding": 0, "resource": ["buffer": 2]]]],
            ["op": "getCurrentTexture", "id": 10, "canvas": "test"],
            ["op": "createTextureView", "id": 11, "texture": 10],
            ["op": "beginRenderPass", "colorAttachments": [[
                "view": 11, "loadOp": "clear", "storeOp": "store",
                "clearValue": ["r": 0, "g": 0, "b": 0, "a": 1],
            ]]],
            ["op": "setPipeline", "pipeline": 3],
            ["op": "setBindGroup", "index": 0, "bindGroup": 5],
            ["op": "draw", "vertexCount": 3],
            ["op": "endPass"],
        ])

        // r = 1+2+3+4 = 10 · g = 5+6+7+8 = 26 · b = 3*10 + 6 = 36.
        // A shifted layout changes **all three** of these.
        try harness.assertPixel(x: 32, y: 32, equals: (10, 26, 36, 255), "integer vec3 layout")
    }

    func test_anIndexedDrawFillsTheQuad() throws {
        let vertices: [Float] = [
            -1, -1, 1, 1, 0,
             1, -1, 1, 1, 0,
             1,  1, 1, 1, 0,
            -1,  1, 1, 1, 0,
        ]
        let indices: [UInt16] = [0, 1, 2, 0, 2, 3]
        let indexData = indices.withUnsafeBufferPointer { Data(buffer: $0).base64EncodedString() }

        harness.executeExpectingSuccess([
            ["op": "configureCanvas", "canvas": "test", "format": "rgba8unorm"],
            ["op": "createShaderModule", "id": 1, "code": Self.triangleShader],
            ["op": "createBuffer", "id": 2, "usage": TestUsage.vertex, "data": vertices.base64],
            ["op": "createBuffer", "id": 3, "usage": TestUsage.index, "data": indexData],
            ["op": "createRenderPipeline", "id": 4, "layout": "auto",
             "vertex": [
                "module": 1, "entryPoint": "vs_main",
                "buffers": [[
                    "arrayStride": 20,
                    "attributes": [
                        ["format": "float32x2", "offset": 0, "shaderLocation": 0],
                        ["format": "float32x3", "offset": 8, "shaderLocation": 1],
                    ],
                ]],
             ],
             "fragment": ["module": 1, "entryPoint": "fs_main", "targets": [["format": "rgba8unorm"]]]],
            ["op": "getCurrentTexture", "id": 10, "canvas": "test"],
            ["op": "createTextureView", "id": 11, "texture": 10],
            ["op": "beginRenderPass", "colorAttachments": [[
                "view": 11, "loadOp": "clear", "storeOp": "store",
                "clearValue": ["r": 0, "g": 0, "b": 0, "a": 1],
            ]]],
            ["op": "setPipeline", "pipeline": 4],
            ["op": "setVertexBuffer", "slot": 0, "buffer": 2],
            ["op": "setIndexBuffer", "buffer": 3, "format": "uint16"],
            ["op": "drawIndexed", "indexCount": 6],
            ["op": "endPass"],
        ])

        // The whole screen must fill with yellow (1,1,0).
        try harness.assertPixel(x: 32, y: 32, equals: (255, 255, 0, 255))
        try harness.assertPixel(x: 2, y: 61, equals: (255, 255, 0, 255))
    }

    func test_alphaBlendingIsApplied() throws {
        let shader = """
        @vertex
        fn vs_main(@builtin(vertex_index) index: u32) -> @builtin(position) vec4f {
            var positions = array<vec2f, 3>(vec2f(-1.0, -1.0), vec2f(3.0, -1.0), vec2f(-1.0, 3.0));
            return vec4f(positions[index], 0.0, 1.0);
        }
        @fragment
        fn fs_main() -> @location(0) vec4f {
            return vec4f(1.0, 0.0, 0.0, 0.5);
        }
        """
        harness.executeExpectingSuccess([
            ["op": "configureCanvas", "canvas": "test", "format": "rgba8unorm"],
            ["op": "createShaderModule", "id": 1, "code": shader],
            ["op": "createRenderPipeline", "id": 2, "layout": "auto",
             "vertex": ["module": 1, "entryPoint": "vs_main"],
             "fragment": ["module": 1, "entryPoint": "fs_main", "targets": [[
                "format": "rgba8unorm",
                "blend": [
                    "color": ["srcFactor": "src-alpha", "dstFactor": "one-minus-src-alpha", "operation": "add"],
                    "alpha": ["srcFactor": "one", "dstFactor": "one-minus-src-alpha", "operation": "add"],
                ],
             ]]]],
            ["op": "getCurrentTexture", "id": 10, "canvas": "test"],
            ["op": "createTextureView", "id": 11, "texture": 10],
            ["op": "beginRenderPass", "colorAttachments": [[
                "view": 11, "loadOp": "clear", "storeOp": "store",
                "clearValue": ["r": 0, "g": 0, "b": 1, "a": 1],
            ]]],
            ["op": "setPipeline", "pipeline": 2],
            ["op": "draw", "vertexCount": 3],
            ["op": "endPass"],
        ])

        // Red at 50% over blue → (0.5, 0, 0.5)
        try harness.assertPixel(x: 32, y: 32, equals: (128, 0, 128, 255), tolerance: 3)
    }

    /// Stacks two layers with **the same blend settings and colors as the demo** (the `blending` scene)
    /// and checks the pixels match what the premultiplied-alpha src-over formula produces.
    ///
    ///   result = src·a + dst·(1 − a)
    ///
    /// Judging "that blend looks off" by eye is easy to get wrong, so it is pinned numerically.
    func test_premultipliedAlphaCompositingMatchesTheFormula() throws {
        let shader = """
        struct Layer { color: vec4f };
        @group(0) @binding(0) var<uniform> layer: Layer;

        @vertex
        fn vs_main(@builtin(vertex_index) index: u32) -> @builtin(position) vec4f {
            var corners = array<vec2f, 3>(vec2f(-1.0, -1.0), vec2f(3.0, -1.0), vec2f(-1.0, 3.0));
            return vec4f(corners[index], 0.0, 1.0);
        }

        @fragment
        fn fs_main() -> @location(0) vec4f {
            // As in the demo, alpha is premultiplied into RGB on output.
            return vec4f(layer.color.rgb * layer.color.a, layer.color.a);
        }
        """
        // The demo's colors and background exactly.
        let background: [Double] = [0.043, 0.055, 0.08]
        let first: [Float] = [1.0, 0.25, 0.3, 0.55]
        let second: [Float] = [0.25, 0.9, 0.45, 0.55]

        harness.executeExpectingSuccess([
            ["op": "configureCanvas", "canvas": "test", "format": "rgba8unorm"],
            ["op": "createShaderModule", "id": 1, "code": shader],
            ["op": "createBuffer", "id": 2, "size": 16,
             "usage": TestUsage.uniform | TestUsage.copyDst, "data": first.base64],
            ["op": "createBuffer", "id": 3, "size": 16,
             "usage": TestUsage.uniform | TestUsage.copyDst, "data": second.base64],
            ["op": "createRenderPipeline", "id": 4, "layout": "auto",
             "vertex": ["module": 1, "entryPoint": "vs_main"],
             "fragment": ["module": 1, "entryPoint": "fs_main", "targets": [[
                "format": "rgba8unorm",
                "blend": [
                    "color": ["srcFactor": "one", "dstFactor": "one-minus-src-alpha", "operation": "add"],
                    "alpha": ["srcFactor": "one", "dstFactor": "one-minus-src-alpha", "operation": "add"],
                ],
             ]]]],
            ["op": "getBindGroupLayout", "id": 5, "pipeline": 4, "index": 0],
            ["op": "createBindGroup", "id": 6, "layout": 5,
             "entries": [["binding": 0, "resource": ["buffer": 2]]]],
            ["op": "createBindGroup", "id": 7, "layout": 5,
             "entries": [["binding": 0, "resource": ["buffer": 3]]]],
            ["op": "getCurrentTexture", "id": 10, "canvas": "test"],
            ["op": "createTextureView", "id": 11, "texture": 10],
            ["op": "beginRenderPass", "colorAttachments": [[
                "view": 11, "loadOp": "clear", "storeOp": "store",
                "clearValue": ["r": background[0], "g": background[1], "b": background[2], "a": 1.0],
            ]]],
            ["op": "setPipeline", "pipeline": 4],
            ["op": "setBindGroup", "index": 0, "bindGroup": 6],
            ["op": "draw", "vertexCount": 3],
            // The second layer goes only over the right half — one pass shows 1-layer and 2-layer together.
            ["op": "setScissorRect", "x": 32, "y": 0, "width": 32, "height": 64],
            ["op": "setBindGroup", "index": 0, "bindGroup": 7],
            ["op": "draw", "vertexCount": 3],
            ["op": "endPass"],
        ])

        /// Computes src-over (premultiplied alpha) on the CPU.
        func over(_ source: [Float], on destination: [Double]) -> [Double] {
            let alpha = Double(source[3])
            return (0..<3).map { Double(source[$0]) * alpha + destination[$0] * (1 - alpha) }
        }
        func bytes(_ color: [Double]) -> (r: Int, g: Int, b: Int, a: Int) {
            (Int((color[0] * 255).rounded()), Int((color[1] * 255).rounded()),
             Int((color[2] * 255).rounded()), 255)
        }

        let oneLayer = over(first, on: background)
        let twoLayers = over(second, on: oneLayer)

        // One layer: red at 55% over the background → (145, 41, 51)
        try harness.assertPixel(x: 16, y: 32, equals: bytes(oneLayer), "one layer")
        // Two layers: green at 55% over that result → (100, 145, 86)
        try harness.assertPixel(x: 48, y: 32, equals: bytes(twoLayers), "two layers")

        // Drawn over an opaque background, so alpha must stay 1 (the canvas must not show through).
        XCTAssertEqual(try harness.pixel(x: 48, y: 32).a, 255)
    }

    // MARK: - Bind groups beyond the layout

    /// The spec ignores a bind group bound beyond the current pipeline layout. Switching from a
    /// 2-group pipeline to a 1-group one in the same pass used to send @group(1) down to a layout with
    /// no Metal index for it — the draw threw, and so did every later draw in the pass. Coming back to
    /// the 2-group pipeline must reuse the group 1 still bound, with no new setBindGroup.
    func test_aGroupBoundBeyondTheLayoutIsIgnoredAndComesBackForALaterPipeline() throws {
        let vertex = """
        @vertex
        fn vs_main(@builtin(vertex_index) index: u32) -> @builtin(position) vec4f {
            var positions = array<vec2f, 3>(vec2f(-1.0, -1.0), vec2f(3.0, -1.0), vec2f(-1.0, 3.0));
            return vec4f(positions[index], 0.0, 1.0);
        }
        """
        let twoGroups = """
        struct Tint { color: vec4f }
        @group(0) @binding(0) var<uniform> base: Tint;
        @group(1) @binding(0) var<uniform> extra: Tint;
        \(vertex)
        @fragment
        fn fs_main() -> @location(0) vec4f { return base.color + extra.color; }
        """
        let oneGroup = """
        struct Tint { color: vec4f }
        @group(0) @binding(0) var<uniform> base: Tint;
        \(vertex)
        @fragment
        fn fs_main() -> @location(0) vec4f { return base.color; }
        """
        let fragmentStage = 0x2

        harness.executeExpectingSuccess([
            ["op": "configureCanvas", "canvas": "test", "format": "rgba8unorm"],
            ["op": "createShaderModule", "id": 1, "code": twoGroups],
            ["op": "createShaderModule", "id": 2, "code": oneGroup],
            ["op": "createBuffer", "id": 5, "size": 16, "usage": TestUsage.uniform | TestUsage.copyDst,
             "data": [Float]([1, 0, 0, 1]).base64],
            ["op": "createBuffer", "id": 6, "size": 16, "usage": TestUsage.uniform | TestUsage.copyDst,
             "data": [Float]([0, 0, 1, 0]).base64],
            // Explicit layouts, so both pipelines accept the same bind groups (auto layouts are exclusive).
            ["op": "createBindGroupLayout", "id": 10, "entries": [
                ["binding": 0, "visibility": fragmentStage, "buffer": ["type": "uniform"]],
            ]],
            ["op": "createPipelineLayout", "id": 12, "bindGroupLayouts": [10, 10]],
            ["op": "createPipelineLayout", "id": 13, "bindGroupLayouts": [10]],
            ["op": "createRenderPipeline", "id": 3, "layout": 12,
             "vertex": ["module": 1, "entryPoint": "vs_main"],
             "fragment": ["module": 1, "entryPoint": "fs_main", "targets": [["format": "rgba8unorm"]]]],
            ["op": "createRenderPipeline", "id": 4, "layout": 13,
             "vertex": ["module": 2, "entryPoint": "vs_main"],
             "fragment": ["module": 2, "entryPoint": "fs_main", "targets": [["format": "rgba8unorm"]]]],
            ["op": "createBindGroup", "id": 20, "layout": 10,
             "entries": [["binding": 0, "resource": ["buffer": 5]]]],
            ["op": "createBindGroup", "id": 21, "layout": 10,
             "entries": [["binding": 0, "resource": ["buffer": 6]]]],
            ["op": "getCurrentTexture", "id": 30, "canvas": "test"],
            ["op": "createTextureView", "id": 31, "texture": 30],
            ["op": "beginRenderPass", "colorAttachments": [[
                "view": 31, "loadOp": "clear", "storeOp": "store",
                "clearValue": ["r": 0, "g": 0, "b": 0, "a": 1],
            ]]],
            // Left third: both groups (red + blue).
            ["op": "setPipeline", "pipeline": 3],
            ["op": "setBindGroup", "index": 0, "bindGroup": 20],
            ["op": "setBindGroup", "index": 1, "bindGroup": 21],
            ["op": "setScissorRect", "x": 0, "y": 0, "width": 21, "height": 64],
            ["op": "draw", "vertexCount": 3],
            // Middle third: the 1-group pipeline — group 1 stays bound but lies beyond its layout.
            ["op": "setPipeline", "pipeline": 4],
            ["op": "setScissorRect", "x": 21, "y": 0, "width": 21, "height": 64],
            ["op": "draw", "vertexCount": 3],
            // Right third: back to the 2-group pipeline with no new setBindGroup.
            ["op": "setPipeline", "pipeline": 3],
            ["op": "setScissorRect", "x": 42, "y": 0, "width": 22, "height": 64],
            ["op": "draw", "vertexCount": 3],
            ["op": "endPass"],
        ])

        try harness.assertPixel(x: 10, y: 32, equals: (255, 0, 255, 255), "two groups")
        try harness.assertPixel(x: 32, y: 32, equals: (255, 0, 0, 255), "one group (group 1 ignored)")
        try harness.assertPixel(x: 53, y: 32, equals: (255, 0, 255, 255), "group 1 back for the 2-group pipeline")
    }

    /// The other side of the line: a layout that **ends** in an empty group still owns that index, so a
    /// non-empty group bound there is incompatible (spec, Dawn) — not "beyond the layout" and ignored.
    func test_aNonEmptyGroupAtATrailingEmptyLayoutSlotIsStillRejected() {
        let shader = """
        struct Tint { color: vec4f }
        @group(0) @binding(0) var<uniform> base: Tint;

        @vertex
        fn vs_main(@builtin(vertex_index) index: u32) -> @builtin(position) vec4f {
            var positions = array<vec2f, 3>(vec2f(-1.0, -1.0), vec2f(3.0, -1.0), vec2f(-1.0, 3.0));
            return vec4f(positions[index], 0.0, 1.0);
        }

        @fragment
        fn fs_main() -> @location(0) vec4f { return base.color; }
        """
        let fragmentStage = 0x2
        let result = harness.execute([
            ["op": "configureCanvas", "canvas": "test", "format": "rgba8unorm"],
            ["op": "createShaderModule", "id": 1, "code": shader],
            ["op": "createBuffer", "id": 5, "size": 16, "usage": TestUsage.uniform | TestUsage.copyDst,
             "data": [Float]([1, 0, 0, 1]).base64],
            ["op": "createBindGroupLayout", "id": 10, "entries": [
                ["binding": 0, "visibility": fragmentStage, "buffer": ["type": "uniform"]],
            ]],
            ["op": "createBindGroupLayout", "id": 11, "entries": [[String: Any]]()],
            ["op": "createPipelineLayout", "id": 12, "bindGroupLayouts": [10, 11]],
            ["op": "createRenderPipeline", "id": 3, "layout": 12,
             "vertex": ["module": 1, "entryPoint": "vs_main"],
             "fragment": ["module": 1, "entryPoint": "fs_main", "targets": [["format": "rgba8unorm"]]]],
            ["op": "createBindGroup", "id": 20, "layout": 10,
             "entries": [["binding": 0, "resource": ["buffer": 5]]]],
            ["op": "getCurrentTexture", "id": 30, "canvas": "test"],
            ["op": "createTextureView", "id": 31, "texture": 30],
            ["op": "beginRenderPass", "colorAttachments": [[
                "view": 31, "loadOp": "clear", "storeOp": "store",
                "clearValue": ["r": 0, "g": 0, "b": 0, "a": 1],
            ]]],
            ["op": "setPipeline", "pipeline": 3],
            ["op": "setBindGroup", "index": 0, "bindGroup": 20],
            ["op": "setBindGroup", "index": 1, "bindGroup": 20],   // non-empty group at the empty slot 1
            ["op": "draw", "vertexCount": 3],
            ["op": "endPass"],
        ])

        XCTAssertEqual(result["ok"] as? Bool, false)
        XCTAssertTrue(harness.describeErrors(result).contains("@group(1)"), harness.describeErrors(result))
    }

    // MARK: - Compute

    func test_aComputeShaderComputesIntoAStorageBuffer() throws {
        let shader = """
        @group(0) @binding(0) var<storage, read> input: array<f32>;
        @group(0) @binding(1) var<storage, read_write> output: array<f32>;

        @compute @workgroup_size(4)
        fn double_values(@builtin(global_invocation_id) id: vec3u) {
            output[id.x] = input[id.x] * 2.0;
        }
        """
        let input: [Float] = [1, 2, 3, 4, 5, 6, 7, 8]

        harness.executeExpectingSuccess([
            ["op": "createShaderModule", "id": 1, "code": shader],
            ["op": "createBuffer", "id": 2, "usage": TestUsage.storage | TestUsage.copyDst,
             "data": input.base64],
            ["op": "createBuffer", "id": 3, "size": 32,
             "usage": TestUsage.storage | TestUsage.copySrc],
            ["op": "createComputePipeline", "id": 4, "layout": "auto",
             "compute": ["module": 1, "entryPoint": "double_values"]],
            ["op": "getBindGroupLayout", "id": 5, "pipeline": 4, "index": 0],
            ["op": "createBindGroup", "id": 6, "layout": 5, "entries": [
                ["binding": 0, "resource": ["buffer": 2]],
                ["binding": 1, "resource": ["buffer": 3]],
            ]],
            ["op": "beginComputePass"],
            ["op": "setPipeline", "pipeline": 4],
            ["op": "setBindGroup", "index": 0, "bindGroup": 6],
            ["op": "dispatchWorkgroups", "x": 2],
            ["op": "endPass"],
        ])

        let output = try harness.readBufferSync(handle: 3, as: Float.self, size: 32)
        XCTAssertEqual(output, [2, 4, 6, 8, 10, 12, 14, 16])
    }

    func test_anOverrideWorkgroupSizeDispatchesThatManyThreads() throws {
        // `@workgroup_size(blockSize)` used to be read as 1 — one workgroup filled one slot, not eight.
        let shader = """
        override blockSize: u32 = 1u;
        @group(0) @binding(0) var<storage, read_write> out: array<u32>;

        @compute @workgroup_size(blockSize)
        fn fill(@builtin(global_invocation_id) id: vec3u) {
            out[id.x] = id.x + 1u;
        }
        """
        harness.executeExpectingSuccess([
            ["op": "createShaderModule", "id": 1, "code": shader],
            ["op": "createBuffer", "id": 2, "size": 32, "usage": TestUsage.storage | TestUsage.copySrc],
            ["op": "createComputePipeline", "id": 3, "layout": "auto",
             "compute": ["module": 1, "entryPoint": "fill", "constants": ["blockSize": 8]]],
            ["op": "getBindGroupLayout", "id": 4, "pipeline": 3, "index": 0],
            ["op": "createBindGroup", "id": 5, "layout": 4, "entries": [
                ["binding": 0, "resource": ["buffer": 2]],
            ]],
            ["op": "beginComputePass"],
            ["op": "setPipeline", "pipeline": 3],
            ["op": "setBindGroup", "index": 0, "bindGroup": 5],
            ["op": "dispatchWorkgroups", "x": 1],
            ["op": "endPass"],
        ])
        let output = try harness.readBufferSync(handle: 2, as: UInt32.self, size: 32)
        XCTAssertEqual(output, [1, 2, 3, 4, 5, 6, 7, 8])

        // Past the device limit it is a pipeline-creation validation error — never a Metal assertion.
        let rejected = harness.execute([
            ["op": "createComputePipeline", "id": 6, "layout": "auto",
             "compute": ["module": 1, "entryPoint": "fill", "constants": ["blockSize": 65536]]],
        ])
        XCTAssertEqual(rejected["ok"] as? Bool, false)
        XCTAssertTrue(
            harness.describeErrors(rejected).contains("exceeds maxComputeWorkgroupSizeX"),
            harness.describeErrors(rejected)
        )
    }

    func test_aZeroWorkgroupDispatchRunsNothing() throws {
        // `dispatchWorkgroups(Math.ceil(n / 64))` with n = 0 is valid and a no-op by the spec. It used to
        // be raised to 1 (running the shader once); handed to Metal as is, the empty grid asserts.
        let shader = """
        @group(0) @binding(0) var<storage, read_write> values: array<f32>;

        @compute @workgroup_size(4)
        fn mark(@builtin(global_invocation_id) id: vec3u) {
            values[id.x] = 7.0;
        }
        """
        let initial: [Float] = [1, 2, 3, 4, 5, 6, 7, 8]
        func dispatch(_ x: Int, _ y: Int = 1, _ z: Int = 1) -> [[String: Any]] {
            [
                ["op": "beginComputePass"],
                ["op": "setPipeline", "pipeline": 3],
                ["op": "setBindGroup", "index": 0, "bindGroup": 5],
                ["op": "dispatchWorkgroups", "x": x, "y": y, "z": z],
                ["op": "endPass"],
            ]
        }

        harness.executeExpectingSuccess([
            ["op": "createShaderModule", "id": 1, "code": shader],
            ["op": "createBuffer", "id": 2, "usage": TestUsage.storage | TestUsage.copySrc,
             "data": initial.base64],
            ["op": "createComputePipeline", "id": 3, "layout": "auto",
             "compute": ["module": 1, "entryPoint": "mark"]],
            ["op": "getBindGroupLayout", "id": 4, "pipeline": 3, "index": 0],
            ["op": "createBindGroup", "id": 5, "layout": 4,
             "entries": [["binding": 0, "resource": ["buffer": 2]]]],
        ] + dispatch(0) + dispatch(2, 0) + dispatch(2, 1, 0))
        XCTAssertEqual(try harness.readBufferSync(handle: 2, as: Float.self, size: 32), initial)

        // `readBuffer` is mapAsync: the buffer stays mapped (and barred from queue work) until unmap.
        harness.executeExpectingSuccess([["op": "unmapBuffer", "buffer": 2]])

        // Past the advertised `maxComputeWorkgroupsPerDimension` is a validation error, not a launch.
        let tooLarge = harness.execute(dispatch(65536))
        XCTAssertEqual(tooLarge["ok"] as? Bool, false)
        XCTAssertTrue(
            harness.describeErrors(tooLarge).contains("maxComputeWorkgroupsPerDimension"),
            harness.describeErrors(tooLarge)
        )
        XCTAssertEqual(try harness.readBufferSync(handle: 2, as: Float.self, size: 32), initial)

        // The control: one workgroup marks exactly its four invocations.
        harness.executeExpectingSuccess([["op": "unmapBuffer", "buffer": 2]] + dispatch(1))
        XCTAssertEqual(
            try harness.readBufferSync(handle: 2, as: Float.self, size: 32), [7, 7, 7, 7, 5, 6, 7, 8]
        )
    }

    func test_arrayLengthReturnsTheBoundSize() throws {
        // A shader cannot know a buffer's size, so the runtime plugs a size table into the reserved index.
        // What is checked here: (1) a whole binding, (2) a partial binding, (3) an array at the end of a struct.
        let shader = """
        struct Particles {
            count: u32,
            items: array<vec4f>,
        };

        @group(0) @binding(0) var<storage, read> whole: array<f32>;
        @group(0) @binding(1) var<storage, read> part: array<vec4f>;
        @group(0) @binding(2) var<storage, read> particles: Particles;
        @group(0) @binding(3) var<storage, read_write> out: array<u32>;

        @compute @workgroup_size(1)
        fn probe() {
            out[0] = arrayLength(&whole);
            out[1] = arrayLength(&part);
            out[2] = arrayLength(&particles.items);
        }
        """
        harness.executeExpectingSuccess([
            ["op": "createShaderModule", "id": 1, "code": shader],
            // 10 f32 = 40 bytes → length 10
            ["op": "createBuffer", "id": 2, "size": 40, "usage": TestUsage.storage],
            // A 96-byte buffer with only 48 bytes bound → 3 vec4f
            ["op": "createBuffer", "id": 3, "size": 96, "usage": TestUsage.storage],
            // count(4) + padding(12) + 2 vec4f(32) = 48 bytes → items length 2
            ["op": "createBuffer", "id": 4, "size": 48, "usage": TestUsage.storage],
            ["op": "createBuffer", "id": 5, "size": 16,
             "usage": TestUsage.storage | TestUsage.copySrc],
            ["op": "createComputePipeline", "id": 6, "layout": "auto",
             "compute": ["module": 1, "entryPoint": "probe"]],
            ["op": "getBindGroupLayout", "id": 7, "pipeline": 6, "index": 0],
            ["op": "createBindGroup", "id": 8, "layout": 7, "entries": [
                ["binding": 0, "resource": ["buffer": 2]],
                ["binding": 1, "resource": ["buffer": 3, "size": 48]],
                ["binding": 2, "resource": ["buffer": 4]],
                ["binding": 3, "resource": ["buffer": 5]],
            ]],
            ["op": "beginComputePass"],
            ["op": "setPipeline", "pipeline": 6],
            ["op": "setBindGroup", "index": 0, "bindGroup": 8],
            ["op": "dispatchWorkgroups", "x": 1],
            ["op": "endPass"],
        ])

        let lengths = try harness.readBufferSync(handle: 5, as: UInt32.self, size: 16)
        XCTAssertEqual(Array(lengths.prefix(3)), [10, 3, 2])
    }

    /// A compute pass that follows a render pass **in the same batch** binds with its own pipeline's layout.
    ///
    /// The backend used to pick the layout as "the render pipeline if one is set, else the compute
    /// one", and the render pipeline survived its pass until the next batch. So the compute pass's
    /// groups were mapped through the render layout — here that layout has no @group(0) @binding(1),
    /// so the dispatch was dropped with a bogus validation error (with matching binding numbers it
    /// would instead have bound the wrong Metal indices silently).
    func test_aComputePassAfterARenderPassInOneBatchUsesTheComputeLayout() throws {
        let renderShader = """
        struct Tint { color: vec4f };
        @group(0) @binding(0) var<uniform> tint: Tint;

        @vertex
        fn vs_main(@builtin(vertex_index) index: u32) -> @builtin(position) vec4f {
            var positions = array<vec2f, 3>(
                vec2f(-1.0, -1.0),
                vec2f( 3.0, -1.0),
                vec2f(-1.0,  3.0),
            );
            return vec4f(positions[index], 0.0, 1.0);
        }

        @fragment
        fn fs_main() -> @location(0) vec4f {
            return tint.color;
        }
        """
        let computeShader = """
        @group(0) @binding(0) var<storage, read> input: array<f32>;
        @group(0) @binding(1) var<storage, read_write> output: array<f32>;

        @compute @workgroup_size(4)
        fn double_values(@builtin(global_invocation_id) id: vec3u) {
            output[id.x] = input[id.x] * 2.0;
        }
        """
        let tint: [Float] = [0.0, 1.0, 0.0, 1.0]   // green
        let input: [Float] = [1, 2, 3, 4, 5, 6, 7, 8]

        harness.executeExpectingSuccess([
            ["op": "configureCanvas", "canvas": "test", "format": "rgba8unorm"],
            ["op": "createShaderModule", "id": 1, "code": renderShader],
            ["op": "createShaderModule", "id": 2, "code": computeShader],
            ["op": "createBuffer", "id": 3, "size": 16,
             "usage": TestUsage.uniform | TestUsage.copyDst, "data": tint.base64],
            ["op": "createBuffer", "id": 4, "usage": TestUsage.storage | TestUsage.copyDst,
             "data": input.base64],
            ["op": "createBuffer", "id": 5, "size": 32,
             "usage": TestUsage.storage | TestUsage.copySrc],
            ["op": "createRenderPipeline", "id": 6, "layout": "auto",
             "vertex": ["module": 1, "entryPoint": "vs_main"],
             "fragment": ["module": 1, "entryPoint": "fs_main", "targets": [["format": "rgba8unorm"]]]],
            ["op": "getBindGroupLayout", "id": 7, "pipeline": 6, "index": 0],
            ["op": "createBindGroup", "id": 8, "layout": 7,
             "entries": [["binding": 0, "resource": ["buffer": 3]]]],
            ["op": "createComputePipeline", "id": 9, "layout": "auto",
             "compute": ["module": 2, "entryPoint": "double_values"]],
            ["op": "getBindGroupLayout", "id": 10, "pipeline": 9, "index": 0],
            ["op": "createBindGroup", "id": 11, "layout": 10, "entries": [
                ["binding": 0, "resource": ["buffer": 4]],
                ["binding": 1, "resource": ["buffer": 5]],
            ]],
            ["op": "getCurrentTexture", "id": 20, "canvas": "test"],
            ["op": "createTextureView", "id": 21, "texture": 20],
            // Render first — leaves a render pipeline behind in the backend.
            ["op": "beginRenderPass", "colorAttachments": [[
                "view": 21, "loadOp": "clear", "storeOp": "store",
                "clearValue": ["r": 0.0, "g": 0.0, "b": 0.0, "a": 1.0],
            ]]],
            ["op": "setPipeline", "pipeline": 6],
            ["op": "setBindGroup", "index": 0, "bindGroup": 8],
            ["op": "draw", "vertexCount": 3],
            ["op": "endPass"],
            // Then compute, in the same batch.
            ["op": "beginComputePass"],
            ["op": "setPipeline", "pipeline": 9],
            ["op": "setBindGroup", "index": 0, "bindGroup": 11],
            ["op": "dispatchWorkgroups", "x": 2],
            ["op": "endPass"],
        ])

        try harness.assertPixel(x: 32, y: 32, equals: (0, 255, 0, 255), "the render pass still draws")
        let output = try harness.readBufferSync(handle: 5, as: Float.self, size: 32)
        XCTAssertEqual(output, [2, 4, 6, 8, 10, 12, 14, 16])
    }

    // MARK: - Textures

    func test_textureSamplingWorks() throws {
        let shader = """
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
            out.uv = positions[index] * 0.5 + vec2f(0.5, 0.5);
            return out;
        }

        @fragment
        fn fs_main(in: Out) -> @location(0) vec4f {
            return textureSample(tex, samp, in.uv);
        }
        """
        // Fill the whole 2x2 texture with magenta (so the result is the same wherever it samples).
        let texels = [UInt8](repeating: 0, count: 16).enumerated().map { index, _ -> UInt8 in
            switch index % 4 {
            case 0: return 255   // R
            case 1: return 0     // G
            case 2: return 255   // B
            default: return 255  // A
            }
        }

        harness.executeExpectingSuccess([
            ["op": "configureCanvas", "canvas": "test", "format": "rgba8unorm"],
            ["op": "createShaderModule", "id": 1, "code": shader],
            ["op": "createTexture", "id": 2, "size": ["width": 2, "height": 2],
             "format": "rgba8unorm", "usage": TestUsage.textureBinding | TestUsage.copyDst],
            ["op": "writeTexture", "texture": 2, "data": Data(texels).base64EncodedString(),
             "size": ["width": 2, "height": 2], "bytesPerRow": 8],
            ["op": "createTextureView", "id": 3, "texture": 2],
            ["op": "createSampler", "id": 4, "magFilter": "linear", "minFilter": "linear"],
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
                "clearValue": ["r": 0, "g": 0, "b": 0, "a": 1],
            ]]],
            ["op": "setPipeline", "pipeline": 5],
            ["op": "setBindGroup", "index": 0, "bindGroup": 7],
            ["op": "draw", "vertexCount": 3],
            ["op": "endPass"],
        ])

        try harness.assertPixel(x: 32, y: 32, equals: (255, 0, 255, 255), "the sampled texel color")
    }

    func test_samplesAnExternalTextureWithEdgeClamping() throws {
        // `textureSampleBaseClampToEdge` pulls the coordinate half a texel inward. So even at uv 0 or 1
        // **the opposite texel does not blend in** — the mechanism that stops video frame edges bleeding.
        // Here a repeat sampler is used deliberately, creating the situation where the opposite side blends without the clamp.
        let shader = """
        @group(0) @binding(0) var frame: texture_external;
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
            // Fill the whole screen with uv 1.0 (the bottom-right corner).
            out.uv = vec2f(1.0, 1.0);
            return out;
        }

        @fragment
        fn fs_main(in: Out) -> @location(0) vec4f {
            return textureSampleBaseClampToEdge(frame, samp, in.uv);
        }
        """
        // 2x2: (0,0) red, the rest blue. With the clamp working at uv=1.0 pure blue comes out;
        // without it, repeat wraps around and a quarter of red mixes in.
        let texels: [UInt8] = [
            255, 0, 0, 255,   0, 0, 255, 255,
            0, 0, 255, 255,   0, 0, 255, 255,
        ]

        harness.executeExpectingSuccess([
            ["op": "configureCanvas", "canvas": "test", "format": "rgba8unorm"],
            ["op": "createShaderModule", "id": 1, "code": shader],
            ["op": "createTexture", "id": 2, "size": ["width": 2, "height": 2],
             "format": "rgba8unorm", "usage": TestUsage.textureBinding | TestUsage.copyDst],
            ["op": "writeTexture", "texture": 2, "data": Data(texels).base64EncodedString(),
             "size": ["width": 2, "height": 2], "bytesPerRow": 8],
            ["op": "createTextureView", "id": 3, "texture": 2],
            ["op": "createSampler", "id": 4, "magFilter": "linear", "minFilter": "linear",
             "addressModeU": "repeat", "addressModeV": "repeat"],
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
                "clearValue": ["r": 0, "g": 0, "b": 0, "a": 1],
            ]]],
            ["op": "setPipeline", "pipeline": 5],
            ["op": "setBindGroup", "index": 0, "bindGroup": 7],
            ["op": "draw", "vertexCount": 3],
            ["op": "endPass"],
        ])

        try harness.assertPixel(x: 32, y: 32, equals: (0, 0, 255, 255), "only the edge texel must come out")
    }

    // MARK: - Depth buffer

    func test_theDepthTestHidesTheTriangleBehind() throws {
        let shader = """
        struct Uniforms { depth: f32, r: f32, g: f32, b: f32 };
        @group(0) @binding(0) var<uniform> u: Uniforms;

        @vertex
        fn vs_main(@builtin(vertex_index) index: u32) -> @builtin(position) vec4f {
            var positions = array<vec2f, 3>(vec2f(-1.0, -1.0), vec2f(3.0, -1.0), vec2f(-1.0, 3.0));
            return vec4f(positions[index], u.depth, 1.0);
        }
        @fragment
        fn fs_main() -> @location(0) vec4f {
            return vec4f(u.r, u.g, u.b, 1.0);
        }
        """
        // Draw front (depth 0.2, red) then back (depth 0.8, green). The depth test must discard the back one.
        let near: [Float] = [0.2, 1, 0, 0]
        let far: [Float] = [0.8, 0, 1, 0]

        harness.executeExpectingSuccess([
            ["op": "configureCanvas", "canvas": "test", "format": "rgba8unorm"],
            ["op": "createShaderModule", "id": 1, "code": shader],
            ["op": "createTexture", "id": 2, "size": ["width": 64, "height": 64],
             "format": "depth32float", "usage": TestUsage.renderAttachment],
            ["op": "createTextureView", "id": 3, "texture": 2],
            ["op": "createBuffer", "id": 4, "size": 16, "usage": TestUsage.uniform, "data": near.base64],
            ["op": "createBuffer", "id": 5, "size": 16, "usage": TestUsage.uniform, "data": far.base64],
            ["op": "createRenderPipeline", "id": 6, "layout": "auto",
             "vertex": ["module": 1, "entryPoint": "vs_main"],
             "fragment": ["module": 1, "entryPoint": "fs_main", "targets": [["format": "rgba8unorm"]]],
             "depthStencil": ["format": "depth32float", "depthWriteEnabled": true, "depthCompare": "less"]],
            ["op": "getBindGroupLayout", "id": 7, "pipeline": 6, "index": 0],
            ["op": "createBindGroup", "id": 8, "layout": 7,
             "entries": [["binding": 0, "resource": ["buffer": 4]]]],
            ["op": "createBindGroup", "id": 9, "layout": 7,
             "entries": [["binding": 0, "resource": ["buffer": 5]]]],
            ["op": "getCurrentTexture", "id": 10, "canvas": "test"],
            ["op": "createTextureView", "id": 11, "texture": 10],
            ["op": "beginRenderPass",
             "colorAttachments": [[
                "view": 11, "loadOp": "clear", "storeOp": "store",
                "clearValue": ["r": 0, "g": 0, "b": 0, "a": 1],
             ]],
             "depthStencilAttachment": [
                "view": 3, "depthClearValue": 1.0, "depthLoadOp": "clear", "depthStoreOp": "store",
             ]],
            ["op": "setPipeline", "pipeline": 6],
            ["op": "setBindGroup", "index": 0, "bindGroup": 8],
            ["op": "draw", "vertexCount": 3],
            ["op": "setBindGroup", "index": 0, "bindGroup": 9],
            ["op": "draw", "vertexCount": 3],
            ["op": "endPass"],
        ])

        try harness.assertPixel(x: 32, y: 32, equals: (255, 0, 0, 255), "the front one (red) must remain")
    }

    // MARK: - Depth bias

    /// Two full-screen triangles at the same depth (0.5): red and green.
    private static let depthBiasShader = """
    @vertex
    fn vs_main(@builtin(vertex_index) index: u32) -> @builtin(position) vec4f {
        var positions = array<vec2f, 3>(vec2f(-1.0, -1.0), vec2f(3.0, -1.0), vec2f(-1.0, 3.0));
        return vec4f(positions[index], 0.5, 1.0);
    }
    @fragment
    fn fs_red() -> @location(0) vec4f {
        return vec4f(1.0, 0.0, 0.0, 1.0);
    }
    @fragment
    fn fs_green() -> @location(0) vec4f {
        return vec4f(0.0, 1.0, 0.0, 1.0);
    }
    """

    /// Pipeline 4 (red) pushes its fragments back with a large `depthBias`; pipeline 5 (green) has none.
    /// The depth buffer is cleared to 0.6 with compare `less`, so at depth 0.5:
    /// - red only passes **without** its bias (0.5 + bias ≥ 0.6 is rejected);
    /// - green only passes **without** a bias (it must not inherit red's).
    /// Drawing red then green therefore leaves green. Black = green inherited the bias, red = the bias never applied.
    private func depthBiasResources() -> [[String: Any]] {
        // Constant bias on a float depth format is scaled by 2^(e - 23) — 2^-24 at depth 0.5 — so this
        // moves red by ~0.6, far past the 0.1 margin. An implementation that clamps to 1.0 rejects it too.
        let depthBias = 10_000_000
        return [
            ["op": "configureCanvas", "canvas": "test", "format": "rgba8unorm"],
            ["op": "createShaderModule", "id": 1, "code": Self.depthBiasShader],
            ["op": "createTexture", "id": 2, "size": ["width": 64, "height": 64],
             "format": "depth32float", "usage": TestUsage.renderAttachment],
            ["op": "createTextureView", "id": 3, "texture": 2],
            ["op": "createRenderPipeline", "id": 4, "layout": "auto",
             "vertex": ["module": 1, "entryPoint": "vs_main"],
             "fragment": ["module": 1, "entryPoint": "fs_red", "targets": [["format": "rgba8unorm"]]],
             "depthStencil": ["format": "depth32float", "depthWriteEnabled": true, "depthCompare": "less",
                              "depthBias": depthBias]],
            ["op": "createRenderPipeline", "id": 5, "layout": "auto",
             "vertex": ["module": 1, "entryPoint": "vs_main"],
             "fragment": ["module": 1, "entryPoint": "fs_green", "targets": [["format": "rgba8unorm"]]],
             "depthStencil": ["format": "depth32float", "depthWriteEnabled": true, "depthCompare": "less"]],
            ["op": "getCurrentTexture", "id": 10, "canvas": "test"],
            ["op": "createTextureView", "id": 11, "texture": 10],
        ]
    }

    private let depthBiasPass: [String: Any] = [
        "op": "beginRenderPass",
        "colorAttachments": [[
            "view": 11, "loadOp": "clear", "storeOp": "store",
            "clearValue": ["r": 0, "g": 0, "b": 0, "a": 1],
        ]],
        "depthStencilAttachment": [
            "view": 3, "depthClearValue": 0.6, "depthLoadOp": "clear", "depthStoreOp": "store",
        ],
    ]

    func test_aPipelineWithoutDepthBiasDoesNotInheritThePreviousPipelinesBias() throws {
        // Metal's depth bias is encoder state that outlives `setRenderPipelineState` — in WebGPU it
        // belongs to the pipeline, so the second pipeline in the same pass must draw unbiased.
        harness.executeExpectingSuccess(depthBiasResources() + [
            depthBiasPass,
            ["op": "setPipeline", "pipeline": 4],
            ["op": "draw", "vertexCount": 3],
            ["op": "setPipeline", "pipeline": 5],
            ["op": "draw", "vertexCount": 3],
            ["op": "endPass"],
        ])

        try harness.assertPixel(
            x: 32, y: 32, equals: (0, 255, 0, 255),
            "green must draw unbiased (black = it inherited red's depthBias, red = the bias never applied)"
        )
    }

    func test_aBundlePipelineWithoutDepthBiasDoesNotInheritThePassPipelinesBias() throws {
        // The same carry-over across a bundle boundary — the replayed `setPipeline` goes down the same path.
        harness.executeExpectingSuccess(depthBiasResources() + [
            ["op": "createRenderBundle", "id": 12, "colorFormats": ["rgba8unorm"],
             "depthStencilFormat": "depth32float",
             "commands": [
                ["op": "setPipeline", "pipeline": 5],
                ["op": "draw", "vertexCount": 3],
             ]],
            depthBiasPass,
            ["op": "setPipeline", "pipeline": 4],
            ["op": "draw", "vertexCount": 3],
            ["op": "executeBundles", "bundles": [12]],
            ["op": "endPass"],
        ])

        try harness.assertPixel(
            x: 32, y: 32, equals: (0, 255, 0, 255),
            "the bundle's green must draw unbiased (black = it inherited the pass pipeline's depthBias)"
        )
    }

    // MARK: - Multisample resolve

    /// `storeOp: "store"` on a resolving attachment keeps the **multisample samples** too — the spec
    /// applies storeOp to the view and resolves on top. Mapping every resolve to Metal's
    /// `.multisampleResolve` discarded them, so a second pass that loads the MSAA texture (the usual
    /// "draw in several passes, resolve each time" pattern) read undefined samples.
    ///
    /// Pass 1 clears the MSAA texture blue, draws red over the left half, stores + resolves. Pass 2
    /// loads it, draws green into the bottom-right quadrant only and resolves again. The canvas holds
    /// pass 2's resolve, so pass 1's red and blue surviving there proves the samples were kept.
    ///
    /// It discriminates on Apple Silicon (tile memory is only written back on store) — on IMR Mac GPUs
    /// (Intel/AMD) the old `.multisampleResolve` may keep the samples anyway.
    func test_aStoringResolvePassKeepsTheMultisampleContentsForALaterLoad() throws {
        try assertStoringResolveKeepsTheMultisampleContents()
    }

    /// The same contract through the emulation used where `.storeAndMultisampleResolve` is missing
    /// (the iOS simulator): store in the pass, resolve in a draw-less pass at its end.
    func test_theEmulatedStoreAndResolveKeepsTheMultisampleContentsForALaterLoad() throws {
        let context = try XCTUnwrap(harness.context, "the emulation switch is a Metal backend hook")
        context.supportsStoreAndMultisampleResolve = false
        try assertStoringResolveKeepsTheMultisampleContents()
    }

    private func assertStoringResolveKeepsTheMultisampleContents(
        file: StaticString = #filePath,
        line: UInt = #line
    ) throws {
        let shader = """
        @vertex
        fn vs_main(@builtin(vertex_index) index: u32) -> @builtin(position) vec4f {
            var positions = array<vec2f, 3>(vec2f(-1.0, -1.0), vec2f(3.0, -1.0), vec2f(-1.0, 3.0));
            return vec4f(positions[index], 0.0, 1.0);
        }
        @fragment
        fn fs_red() -> @location(0) vec4f {
            return vec4f(1.0, 0.0, 0.0, 1.0);
        }
        @fragment
        fn fs_green() -> @location(0) vec4f {
            return vec4f(0.0, 1.0, 0.0, 1.0);
        }
        """
        func pipeline(id: Int, entryPoint: String) -> [String: Any] {
            [
                "op": "createRenderPipeline", "id": id, "layout": "auto",
                "vertex": ["module": 1, "entryPoint": "vs_main"],
                "fragment": ["module": 1, "entryPoint": entryPoint, "targets": [["format": "rgba8unorm"]]],
                "multisample": ["count": 4],
            ]
        }

        harness.executeExpectingSuccess([
            ["op": "configureCanvas", "canvas": "test", "format": "rgba8unorm"],
            ["op": "createShaderModule", "id": 1, "code": shader],
            ["op": "createTexture", "id": 2, "size": ["width": 64, "height": 64],
             "format": "rgba8unorm", "sampleCount": 4, "usage": TestUsage.renderAttachment],
            ["op": "createTextureView", "id": 3, "texture": 2],
            pipeline(id: 4, entryPoint: "fs_red"),
            pipeline(id: 5, entryPoint: "fs_green"),
            ["op": "getCurrentTexture", "id": 10, "canvas": "test"],
            ["op": "createTextureView", "id": 11, "texture": 10],

            ["op": "beginRenderPass", "colorAttachments": [[
                "view": 3, "resolveTarget": 11, "loadOp": "clear", "storeOp": "store",
                "clearValue": ["r": 0, "g": 0, "b": 1, "a": 1],
            ]]],
            ["op": "setPipeline", "pipeline": 4],
            ["op": "setScissorRect", "x": 0, "y": 0, "width": 32, "height": 64],
            ["op": "draw", "vertexCount": 3],
            ["op": "endPass"],

            ["op": "beginRenderPass", "colorAttachments": [[
                "view": 3, "resolveTarget": 11, "loadOp": "load", "storeOp": "discard",
            ]]],
            ["op": "setPipeline", "pipeline": 5],
            ["op": "setScissorRect", "x": 32, "y": 32, "width": 32, "height": 32],
            ["op": "draw", "vertexCount": 3],
            ["op": "endPass"],
        ], file: file, line: line)

        try harness.assertPixel(x: 16, y: 16, equals: (255, 0, 0, 255), "pass 1's red must survive the load",
                                file: file, line: line)
        try harness.assertPixel(x: 16, y: 48, equals: (255, 0, 0, 255), "pass 1's red must survive the load",
                                file: file, line: line)
        try harness.assertPixel(x: 48, y: 16, equals: (0, 0, 255, 255), "pass 1's clear must survive the load",
                                file: file, line: line)
        try harness.assertPixel(x: 48, y: 48, equals: (0, 255, 0, 255), "pass 2's own draw",
                                file: file, line: line)
    }

    // MARK: - HDR readback

    func test_anRGBA16FloatSurfaceDoesNotLoseValuesOutsideSDR() throws {
        // Inside the triangle is what the fragment wrote, outside is the clear value — both go past 1.0.
        // On an 8-bit surface everything here would clip to 1.0.
        let shader = """
        @vertex
        fn vs_main(@builtin(vertex_index) index: u32) -> @builtin(position) vec4f {
            var positions = array<vec2f, 3>(vec2f(-0.5, -0.5), vec2f(0.5, -0.5), vec2f(0.0, 0.5));
            return vec4f(positions[index], 0.0, 1.0);
        }
        @fragment
        fn fs_main() -> @location(0) vec4f {
            return vec4f(2.5, 0.5, -0.25, 1.0);
        }
        """

        harness.executeExpectingSuccess([
            ["op": "configureCanvas", "canvas": "test", "format": "rgba16float"],
            ["op": "createShaderModule", "id": 1, "code": shader],
            ["op": "createRenderPipeline", "id": 2, "layout": "auto",
             "vertex": ["module": 1, "entryPoint": "vs_main"],
             "fragment": ["module": 1, "entryPoint": "fs_main", "targets": [["format": "rgba16float"]]]],
            ["op": "getCurrentTexture", "id": 10, "canvas": "test"],
            ["op": "createTextureView", "id": 11, "texture": 10],
            ["op": "beginRenderPass", "colorAttachments": [[
                "view": 11, "loadOp": "clear", "storeOp": "store",
                "clearValue": ["r": 4.0, "g": 0.0, "b": 0.0, "a": 1.0],
            ]]],
            ["op": "setPipeline", "pipeline": 2],
            ["op": "draw", "vertexCount": 3],
            ["op": "endPass"],
        ])

        let readback = try harness.readback()
        XCTAssertEqual(readback.format, .rgba16float)
        XCTAssertEqual(readback.bytesPerRow, 64 * 8, "it must be 8 bytes per pixel")
        XCTAssertEqual(readback.data.count, 64 * 64 * 8)

        try harness.assertPixelFloat(
            x: 32, y: 32, equals: SIMD4<Float>(2.5, 0.5, -0.25, 1),
            "inside the triangle — values above 1.0 and negatives must survive"
        )
        try harness.assertPixelFloat(
            x: 1, y: 1, equals: SIMD4<Float>(4, 0, 0, 1), "the clear value must not clip either"
        )
    }

    // MARK: - Global shadowing (whether scope resolution is right down to the value)

    func test_theScopeOfALocalShadowingAGlobalIsCorrectAtRuntime() throws {
        // base before the declaration is the global (green); after it, the local (red). A mistake in scope
        // resolution or rename substitution yields red — something a successful compile cannot catch.
        let shader = """
        var<private> base : vec4f;

        fn shade() -> vec4f {
            let inherited = base;
            var base : vec4f = vec4f(1.0, 0.0, 0.0, 1.0);
            return inherited + base * 0.0;
        }

        @vertex
        fn vs_main(@builtin(vertex_index) index: u32) -> @builtin(position) vec4f {
            var positions = array<vec2f, 3>(
                vec2f(-1.0, -1.0),
                vec2f( 3.0, -1.0),
                vec2f(-1.0,  3.0),
            );
            return vec4f(positions[index], 0.0, 1.0);
        }

        @fragment
        fn fs_main() -> @location(0) vec4f {
            base = vec4f(0.0, 1.0, 0.0, 1.0);
            return shade();
        }
        """

        harness.executeExpectingSuccess([
            ["op": "configureCanvas", "canvas": "test", "format": "rgba8unorm"],
            ["op": "createShaderModule", "id": 1, "code": shader],
            ["op": "createRenderPipeline", "id": 2, "layout": "auto",
             "vertex": ["module": 1, "entryPoint": "vs_main"],
             "fragment": ["module": 1, "entryPoint": "fs_main", "targets": [["format": "rgba8unorm"]]]],
            ["op": "getCurrentTexture", "id": 10, "canvas": "test"],
            ["op": "createTextureView", "id": 11, "texture": 10],
            ["op": "beginRenderPass", "colorAttachments": [[
                "view": 11, "loadOp": "clear", "storeOp": "store",
                "clearValue": ["r": 0.0, "g": 0.0, "b": 0.0, "a": 1.0],
            ]]],
            ["op": "setPipeline", "pipeline": 2],
            ["op": "draw", "vertexCount": 3],
            ["op": "endPass"],
        ])

        try harness.assertPixel(x: 32, y: 32, equals: (0, 255, 0, 255), "a reference before the declaration must see the global")
    }
}
