import XCTest
import LynxWebGPUCore
@testable import LynxWebGPUShader

/// The transformations filling **the places WGSL defines but MSL (C++) leaves undefined**.
///
/// All of them are behaviour the spec requires, and without them a driver may do anything —
/// read or overwrite adjacent memory, or have the optimizer treat it as "cannot happen" and delete
/// the surrounding code. Tint (Dawn's WGSL compiler) does the same under the same names.
///
/// Each case checks both (1) the expected MSL fragment and (2) **passing the real Metal compiler**.
final class ShaderSafetyTests: XCTestCase {
    private func translate(
        _ source: String,
        entryPoints: [String],
        file: StaticString = #filePath,
        line: UInt = #line
    ) throws -> String {
        let module = try WGSLShaderModule(source: source)
        let groups = module.autoBindGroupLayouts(entryPoints: entryPoints)
        let bindings = try WGSLBindingAssigner.assign(groups: groups)
        let msl = try module.translateToMSL(entryPoints: entryPoints, bindings: bindings)
        MetalCompilerHarness.assertCompiles(msl, file: file, line: line)
        return msl
    }

    // MARK: - Index range (robustness)

    /// A fixed-size array **knows its size in the type** — a C++ template extracts the bound.
    func test_fixedSizeArrayIndexingIsClampedToRange() throws {
        let msl = try translate("""
        @group(0) @binding(0) var<storage, read> data: array<f32, 8>;
        @group(0) @binding(1) var<uniform> idx: u32;
        @fragment fn fs() -> @location(0) vec4f {
            return vec4f(data[idx], 0, 0, 1);
        }
        """, entryPoints: ["fs"])
        XCTAssertTrue(msl.contains("wgpu_at(data, idx)"), "indexing does not go through the clamp\n\(msl)")
    }

    /// A runtime-sized array has no size in its type — the bound comes from the **buffer size table**.
    /// The index comes from a uniform, so it is ultimately a value the bundle (JS) decides.
    func test_runtimeSizedArrayIndexingIsClampedByBufferSize() throws {
        let msl = try translate("""
        @group(0) @binding(0) var<storage, read_write> data: array<f32>;
        @group(0) @binding(1) var<uniform> idx: u32;
        @compute @workgroup_size(1) fn cs() {
            data[idx] = 1.0;
        }
        """, entryPoints: ["cs"])
        XCTAssertTrue(msl.contains("wgpu_store_n(data, idx"), "a runtime array write does not go through the clamp\n\(msl)")
        XCTAssertTrue(msl.contains("wgpu_buffer_sizes"), "no size table to take the bound from\n\(msl)")
    }

    /// Once the size table is required, **the pipeline must reach the same answer** — a mismatch makes
    /// the shader read an unbound buffer.
    func test_indexingARuntimeArrayAloneReportsTheSizeTableAsNeeded() throws {
        let module = try WGSLShaderModule(source: """
        @group(0) @binding(0) var<storage, read_write> data: array<f32>;
        @compute @workgroup_size(1) fn cs() { data[0] = 1.0; }
        """)
        XCTAssertTrue(
            module.usesArrayLength(entryPoints: ["cs"]),
            "indexing needs the table even without arrayLength()"
        )
    }

    /// Vector components cannot bind to a reference (an MSL limit), so reads go by value and writes through a store helper.
    func test_vectorComponentIndexingIsClampedForBothReadAndWrite() throws {
        let msl = try translate("""
        @group(0) @binding(0) var<uniform> idx: u32;
        @fragment fn fs() -> @location(0) vec4f {
            var v = vec4f(1, 2, 3, 4);
            v[idx] = 9.0;
            return vec4f(v[idx], 0, 0, 1);
        }
        """, entryPoints: ["fs"])
        XCTAssertTrue(msl.contains("wgpu_store(v, idx"), "a vector write does not go through the clamp\n\(msl)")
        XCTAssertTrue(msl.contains("wgpu_at(v, idx)"), "a vector read does not go through the clamp\n\(msl)")
    }

    func test_matrixColumnIndexingAndNestedIndexingCompile() throws {
        _ = try translate("""
        @group(0) @binding(0) var<uniform> idx: u32;
        @fragment fn fs() -> @location(0) vec4f {
            var m = mat4x4f();
            m[idx][idx] = 1.0;
            var grid = array<array<f32, 4>, 4>();
            grid[idx][idx] = 2.0;
            return vec4f(m[idx][idx] + grid[idx][idx], 0, 0, 1);
        }
        """, entryPoints: ["fs"])
    }

    /// A compound assignment (`+=`) must take the same path — guarding one side only lets it leak through the other.
    func test_compoundAssignmentGoesThroughTheClampToo() throws {
        let msl = try translate("""
        @group(0) @binding(0) var<storage, read_write> data: array<f32, 4>;
        @group(0) @binding(1) var<uniform> idx: u32;
        @compute @workgroup_size(1) fn cs() {
            data[idx] += 1.0;
        }
        """, entryPoints: ["cs"])
        XCTAssertTrue(msl.contains("wgpu_store(data, idx"), "a compound assignment does not go through the clamp\n\(msl)")
    }

    // MARK: - Integer division, shifts and conversion

    /// WGSL defines `x / 0 == x` and `x % 0 == 0`. In C++ both are UB.
    func test_integerDivisionAndRemainderFilterAZeroDenominator() throws {
        let msl = try translate("""
        @group(0) @binding(0) var<uniform> d: i32;
        @fragment fn fs() -> @location(0) vec4f {
            return vec4f(f32(100 / d), f32(100 % d), 0, 1);
        }
        """, entryPoints: ["fs"])
        XCTAssertTrue(msl.contains("wgpu_div(100, d)"), "division has no guard\n\(msl)")
        XCTAssertTrue(msl.contains("wgpu_mod(100, d)"), "remainder has no guard\n\(msl)")
    }

    /// A shift at or beyond the width is UB in C++ — only the low 5 bits are kept (the same rule as Tint).
    func test_theShiftAmountIsMasked() throws {
        let msl = try translate("""
        @group(0) @binding(0) var<uniform> s: u32;
        @fragment fn fs() -> @location(0) vec4f {
            return vec4f(f32(1u << s), f32(256u >> s), 0, 1);
        }
        """, entryPoints: ["fs"])
        XCTAssertTrue(msl.contains("wgpu_shl(1u, s)"), "the left shift is not masked\n\(msl)")
        XCTAssertTrue(msl.contains("wgpu_shr(256u, s)"), "the right shift is not masked\n\(msl)")
    }

    /// WGSL defines an out-of-range float → int conversion as **saturating**. A C++ cast is UB.
    func test_floatToIntegerConversionSaturates() throws {
        let msl = try translate("""
        @group(0) @binding(0) var<uniform> f: f32;
        @fragment fn fs() -> @location(0) vec4f {
            return vec4f(f32(i32(f)), f32(u32(f)), f32(vec2i(f).x), 1);
        }
        """, entryPoints: ["fs"])
        XCTAssertTrue(msl.contains("wgpu_ftoi(f)"), "the i32() conversion does not saturate\n\(msl)")
        XCTAssertTrue(msl.contains("wgpu_ftou(f)"), "the u32() conversion does not saturate\n\(msl)")
        XCTAssertTrue(msl.contains("wgpu_ftoi_n<2>(f)"), "the vector conversion does not saturate\n\(msl)")
    }

    /// Moving an integer vector into another integer vector has nothing to do with saturation — it must not break trying to narrow.
    func test_integerVectorConversionPassesThrough() throws {
        _ = try translate("""
        @group(0) @binding(0) var<uniform> v: vec2u;
        @fragment fn fs() -> @location(0) vec4f {
            let signed = vec2i(v);
            return vec4f(f32(signed.x), 0, 0, 1);
        }
        """, entryPoints: ["fs"])
    }

    // MARK: - workgroup zero-initialization

    /// WGSL guarantees zero-initialization of `var<workgroup>`. MSL's threadgroup storage does not —
    /// without it you read **the leftovers of a previous dispatch**.
    func test_workgroupVariablesAreZeroedWithABarrier() throws {
        let msl = try translate("""
        var<workgroup> tile: array<f32, 64>;
        var<workgroup> total: f32;
        @group(0) @binding(0) var<storage, read_write> out: array<f32>;
        @compute @workgroup_size(64)
        fn cs(@builtin(local_invocation_id) lid: vec3u) {
            out[lid.x] = tile[lid.x] + total;
        }
        """, entryPoints: ["cs"])
        XCTAssertTrue(msl.contains("wgpu_zi"), "the array is not zeroed\n\(msl)")
        XCTAssertTrue(msl.contains("total = float{}"), "the scalar is not zeroed\n\(msl)")
        XCTAssertTrue(
            msl.contains("threadgroup_barrier(mem_flags::mem_threadgroup)"),
            "without the barrier another thread reads a slot not yet zeroed\n\(msl)"
        )
    }

    /// A shader already taking `local_invocation_index` must not get **the same attribute twice**.
    func test_anIndexBuiltinAlreadyTakenIsNotDuplicated() throws {
        let msl = try translate("""
        var<workgroup> tile: array<f32, 8>;
        @group(0) @binding(0) var<storage, read_write> out: array<f32>;
        @compute @workgroup_size(8)
        fn cs(@builtin(local_invocation_index) i: u32) {
            out[i] = tile[i];
        }
        """, entryPoints: ["cs"])
        let occurrences = msl.components(separatedBy: "[[thread_index_in_threadgroup]]").count - 1
        XCTAssertEqual(occurrences, 1, "the same builtin was attached twice\n\(msl)")
    }

    func test_workgroupAtomicsAreZeroedWithStore() throws {
        let msl = try translate("""
        var<workgroup> counter: atomic<u32>;
        @group(0) @binding(0) var<storage, read_write> out: array<u32>;
        @compute @workgroup_size(4)
        fn cs(@builtin(local_invocation_id) lid: vec3u) {
            atomicAdd(&counter, 1u);
            workgroupBarrier();
            out[lid.x] = atomicLoad(&counter);
        }
        """, entryPoints: ["cs"])
        XCTAssertTrue(
            msl.contains("atomic_store_explicit(&counter, 0, memory_order_relaxed)"),
            "an atomic is zeroed only by store, not assignment\n\(msl)"
        )
    }

    // MARK: - Writes after discard

    /// MSL's `discard_fragment()` is not an immediate return — the code after it keeps running and
    /// **the discarded fragment corrupts storage.**
    func test_storageWritesAfterDiscardAreMasked() throws {
        let msl = try translate("""
        @group(0) @binding(0) var<storage, read_write> out: array<f32>;
        @group(0) @binding(1) var<uniform> cutoff: f32;
        @fragment fn fs(@builtin(position) p: vec4f) -> @location(0) vec4f {
            if (p.x > cutoff) { discard; }
            out[0] = 1.0;
            return vec4f(1);
        }
        """, entryPoints: ["fs"])
        XCTAssertTrue(msl.contains("wgpu_discarded = true"), "discard did not become a flag\n\(msl)")
        XCTAssertTrue(msl.contains("if (!wgpu_discarded)"), "the write is not masked\n\(msl)")
        XCTAssertTrue(
            msl.contains("if (wgpu_discarded) { discard_fragment(); }"),
            "the real discard must happen at the end of the entry point (so derivatives in the same quad stay alive)\n\(msl)"
        )
    }

    /// A shader with no `discard` must carry **no cost at all**.
    func test_withNoDiscardNoFlagIsCreated() throws {
        let msl = try translate("""
        @group(0) @binding(0) var<storage, read_write> out: array<f32>;
        @fragment fn fs() -> @location(0) vec4f {
            out[0] = 1.0;
            return vec4f(1);
        }
        """, entryPoints: ["fs"])
        XCTAssertFalse(msl.contains("wgpu_discarded"), "a flag appeared with no discard present\n\(msl)")
    }

    /// A `discard` inside a helper must see the same flag — the flag threads down the call graph.
    func test_aDiscardInAHelperUsesTheSameFlag() throws {
        let msl = try translate("""
        @group(0) @binding(0) var<storage, read_write> out: array<f32>;
        fn maybeDiscard(x: f32) { if (x > 0.5) { discard; } }
        @fragment fn fs(@builtin(position) p: vec4f) -> @location(0) vec4f {
            maybeDiscard(p.x);
            out[0] = 1.0;
            return vec4f(1);
        }
        """, entryPoints: ["fs"])
        XCTAssertTrue(
            msl.contains("thread bool& wgpu_discarded"),
            "the flag does not thread down into the helper\n\(msl)"
        )
    }

    /// A user global spelled like the flag stays a separate variable. It used to share the flag's name
    /// in the emitter's global table, which **trapped** (killing the host process) on the duplicate.
    func test_aUserGlobalNamedLikeTheDiscardFlagStaysSeparate() throws {
        let msl = try translate("""
        @group(0) @binding(0) var<storage, read_write> out: array<f32>;
        var<private> wgpu_discarded: f32 = 2.0;
        @fragment fn fs(@builtin(position) p: vec4f) -> @location(0) vec4f {
            if (p.x > wgpu_discarded) { discard; }
            out[0] = wgpu_discarded;
            return vec4f(1);
        }
        """, entryPoints: ["fs"])
        XCTAssertTrue(msl.contains("float wgpu_id_wgpu_discarded = 2.0;"), "the user global was lost\n\(msl)")
        XCTAssertTrue(msl.contains("bool wgpu_discarded{};"), "the flag was lost\n\(msl)")
        XCTAssertTrue(msl.contains("p.x > wgpu_id_wgpu_discarded"), "\(msl)")
        XCTAssertTrue(msl.contains("wgpu_discarded = true"), "discard did not become a flag\n\(msl)")
        XCTAssertTrue(
            msl.contains("if (wgpu_discarded) { discard_fragment(); }"),
            "the real discard must happen at the end of the entry point\n\(msl)"
        )
    }

    // MARK: - Infinite loops

    /// An infinite loop is UB in C++ — the compiler may assume it ends and delete the surrounding code.
    func test_anInfiniteLoopGainsOneMoreExitCondition() throws {
        let msl = try translate("""
        @group(0) @binding(0) var<uniform> n: u32;
        @fragment fn fs() -> @location(0) vec4f {
            var i = 0u;
            loop {
                if (i >= n) { break; }
                i = i + 1u;
            }
            return vec4f(f32(i), 0, 0, 1);
        }
        """, entryPoints: ["fs"])
        XCTAssertTrue(msl.contains("wgpu_loop_guard_1"), "the infinite loop has no guard\n\(msl)")
        XCTAssertTrue(msl.contains("break;"), "the guard does not lead to an exit\n\(msl)")
    }

    /// A `for`/`while` with a condition gives the compiler grounds to see an end — no guard is attached.
    func test_aLoopWithAConditionGetsNoGuard() throws {
        let msl = try translate("""
        @fragment fn fs() -> @location(0) vec4f {
            var total = 0.0;
            for (var i = 0u; i < 4u; i = i + 1u) { total = total + 1.0; }
            return vec4f(total, 0, 0, 1);
        }
        """, entryPoints: ["fs"])
        XCTAssertFalse(msl.contains("wgpu_loop_guard"), "an unnecessary guard was attached to a loop with a condition\n\(msl)")
    }

    func test_guardNamesOfNestedInfiniteLoopsDoNotCollide() throws {
        let msl = try translate("""
        @group(0) @binding(0) var<uniform> n: u32;
        @fragment fn fs() -> @location(0) vec4f {
            var i = 0u;
            loop {
                if (i >= n) { break; }
                var j = 0u;
                loop {
                    if (j >= n) { break; }
                    j = j + 1u;
                }
                i = i + 1u;
            }
            return vec4f(f32(i), 0, 0, 1);
        }
        """, entryPoints: ["fs"])
        XCTAssertTrue(msl.contains("wgpu_loop_guard_1"), "\(msl)")
        XCTAssertTrue(msl.contains("wgpu_loop_guard_2"), "guard names of nested loops collide\n\(msl)")
    }

    // MARK: - Builtin argument counts

    /// The shader source comes from the bundle, so a builtin with missing arguments is **untrusted input**.
    /// Lowerings that pick arguments out by position used to index past the end — a Swift trap that
    /// killed the host process. It must be an ordinary compile error instead.
    private func assertRejected(
        _ source: String,
        entryPoints: [String],
        mentioning callee: String,
        file: StaticString = #filePath,
        line: UInt = #line
    ) {
        do {
            let module = try WGSLShaderModule(source: source)
            let bindings = try WGSLBindingAssigner.assign(groups: module.autoBindGroupLayouts(entryPoints: entryPoints))
            XCTAssertThrowsError(
                try module.translateToMSL(entryPoints: entryPoints, bindings: bindings), file: file, line: line
            ) { error in
                let error = error as? WGPUError
                XCTAssertEqual(error?.kind, .validation, "\(String(describing: error))", file: file, line: line)
                XCTAssertTrue(
                    error?.message.contains("\(callee)()") == true,
                    "the error must name the builtin: \(String(describing: error))", file: file, line: line
                )
            }
        } catch {
            XCTFail("must parse and fail at translation, not \(error)", file: file, line: line)
        }
    }

    func test_atomicsAndQuantizeWithMissingArgumentsAreCompileErrors() {
        let header = """
        @group(0) @binding(0) var<storage, read_write> counter: atomic<u32>;
        @group(0) @binding(1) var<storage, read_write> out: array<f32>;
        """
        let calls = [
            ("atomicLoad", "out[0] = f32(atomicLoad());"),
            ("atomicStore", "atomicStore(&counter);"),
            ("atomicExchange", "out[0] = f32(atomicExchange(&counter));"),
            ("atomicAdd", "out[0] = f32(atomicAdd(&counter));"),
            ("quantizeToF16", "out[0] = quantizeToF16();"),
        ]
        for (callee, statement) in calls {
            assertRejected("""
            \(header)
            @compute @workgroup_size(1) fn cs() { \(statement) }
            """, entryPoints: ["cs"], mentioning: callee)
        }
    }

    /// Every builtin in the arity table, one short and one long — the table is the only guard in
    /// front of the positional lowerings, so an entry that is too permissive shows up here as a trap.
    func test_everyPositionalBuiltinRejectsAWrongArgumentCount() {
        for (callee, arity) in MSLTypeMapping.builtinArities.sorted(by: { $0.key < $1.key }) {
            var counts = [arity.upperBound + 1]
            if arity.lowerBound > 0 { counts.append(arity.lowerBound - 1) }
            for count in counts {
                let arguments = Array(repeating: "1.0", count: count).joined(separator: ", ")
                assertRejected("""
                @compute @workgroup_size(1) fn cs() { \(callee)(\(arguments)); }
                """, entryPoints: ["cs"], mentioning: callee)
            }
        }
    }

    func test_wellFormedAtomicCallsStillTranslate() throws {
        let msl = try translate("""
        @group(0) @binding(0) var<storage, read_write> counter: atomic<u32>;
        @group(0) @binding(1) var<storage, read_write> out: array<u32>;
        @compute @workgroup_size(1) fn cs() {
            atomicStore(&counter, 1u);
            let added = atomicAdd(&counter, 2u);
            let swapped = atomicExchange(&counter, 3u);
            out[0] = atomicLoad(&counter) + added + swapped;
        }
        """, entryPoints: ["cs"])
        XCTAssertTrue(msl.contains("atomic_store_explicit(&counter, 1u, memory_order_relaxed)"), msl)
        XCTAssertTrue(msl.contains("atomic_fetch_add_explicit(&counter, 2u, memory_order_relaxed)"), msl)
        XCTAssertTrue(msl.contains("atomic_exchange_explicit(&counter, 3u, memory_order_relaxed)"), msl)
        XCTAssertTrue(msl.contains("atomic_load_explicit(&counter, memory_order_relaxed)"), msl)
    }

    func test_wellFormedQuantizeAndBitcastStillTranslate() throws {
        let msl = try translate("""
        @fragment fn fs() -> @location(0) vec4f {
            let q = quantizeToF16(1.5);
            let bits = bitcast<f32>(1065353216u);
            return vec4f(q, bits, 0.0, 1.0);
        }
        """, entryPoints: ["fs"])
        XCTAssertTrue(msl.contains("float(half(1.5))"), msl)
        XCTAssertTrue(msl.contains("as_type<float>(1065353216u)"), msl)
    }

    func test_wellFormedTextureCallsAtTheirLowerArityBoundsStillTranslate() throws {
        let msl = try translate("""
        @group(0) @binding(0) var tex: texture_2d<f32>;
        @group(0) @binding(1) var samp: sampler;
        @group(0) @binding(2) var depth: texture_depth_2d;
        @group(0) @binding(3) var cmp: sampler_comparison;
        @fragment fn fs(@location(0) uv: vec2f) -> @location(0) vec4f {
            let biased = textureSampleBias(tex, samp, uv, 0.5);
            let graded = textureSampleGrad(tex, samp, uv, vec2f(0.01, 0.0), vec2f(0.0, 0.01));
            let shadow = textureSampleCompare(depth, cmp, uv, 0.5);
            let levels = textureNumLevels(tex);
            let size = textureDimensions(tex, 0);
            return biased + graded + vec4f(shadow, f32(levels), f32(size.x), 1.0);
        }
        """, entryPoints: ["fs"])
        XCTAssertTrue(msl.contains("tex.sample(samp, uv, bias(0.5))"), msl)
        XCTAssertTrue(msl.contains("tex.sample(samp, uv, gradient2d("), msl)
        XCTAssertTrue(msl.contains("depth.sample_compare(cmp, uv, 0.5)"), msl)
        XCTAssertTrue(msl.contains("tex.get_num_mip_levels()"), msl)
    }

    /// The table's risk is at the **upper** bounds — an arrayed texture plus an offset is the longest form of each family.
    /// The sampling assertions stop before the closing parenthesis on purpose: the emitter accepts the trailing
    /// `offset` but does not emit it yet, and this test pins the arity bounds, not that omission.
    func test_wellFormedTextureCallsAtTheirUpperArityBoundsStillTranslate() throws {
        let msl = try translate("""
        @group(0) @binding(0) var arr: texture_2d_array<f32>;
        @group(0) @binding(1) var samp: sampler;
        @group(0) @binding(2) var depths: texture_depth_2d_array;
        @group(0) @binding(3) var cmp: sampler_comparison;
        @fragment fn fs(@location(0) uv: vec2f) -> @location(0) vec4f {
            let a = textureSample(arr, samp, uv, 0, vec2i(1, 1));
            let b = textureSampleBias(arr, samp, uv, 0, 0.5, vec2i(1, 1));
            let c = textureSampleLevel(arr, samp, uv, 0, 0.0, vec2i(1, 1));
            let d = textureSampleGrad(arr, samp, uv, 0, vec2f(0.01, 0.0), vec2f(0.0, 0.01), vec2i(1, 1));
            let e = textureSampleCompare(depths, cmp, uv, 0, 0.5, vec2i(1, 1));
            let f = textureSampleCompareLevel(depths, cmp, uv, 0, 0.5, vec2i(1, 1));
            let g = textureLoad(arr, vec2i(0, 0), 0, 0);
            return a + b + c + d + g + vec4f(e, f, 0.0, 1.0);
        }
        """, entryPoints: ["fs"])
        XCTAssertTrue(msl.contains("arr.sample(samp, uv, 0"), msl)
        XCTAssertTrue(msl.contains("arr.sample(samp, uv, 0, bias(0.5)"), msl)
        XCTAssertTrue(msl.contains("arr.sample(samp, uv, 0, level(0.0)"), msl)
        XCTAssertTrue(msl.contains("arr.sample(samp, uv, 0, gradient2d("), msl)
        XCTAssertTrue(msl.contains("depths.sample_compare(cmp, uv, 0, 0.5"), msl)
        XCTAssertTrue(msl.contains("arr.read(uint2("), msl)

        let store = try translate("""
        @group(0) @binding(0) var layers: texture_storage_2d_array<rgba8unorm, write>;
        @compute @workgroup_size(1) fn cs() {
            textureStore(layers, vec2i(0, 0), 1, vec4f(1.0, 0.0, 0.0, 1.0));
        }
        """, entryPoints: ["cs"])
        XCTAssertTrue(store.contains("layers.write("), store)
    }
}
