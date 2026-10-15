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

    /// A runtime-sized array reached **through a `let` pointer** is clamped by the same table.
    ///
    /// Valid WGSL, and the index is a bundle-chosen value: left on the plain pointer path,
    /// `(*p)[100000000u] = 7u` would write 400 MB past the binding.
    func test_aRuntimeArrayReachedThroughALetPointerIsClamped() throws {
        let msl = try translate("""
        @group(0) @binding(0) var<storage, read_write> data: array<u32>;
        @group(0) @binding(1) var<uniform> idx: u32;
        @compute @workgroup_size(1) fn cs() {
            let p = &data;
            let alsoP = p;
            (*p)[100000000u] = 7u;
            (*alsoP)[idx] += (*p)[idx];
        }
        """, entryPoints: ["cs"])
        let count = "(wgpu_buffer_sizes[0] / uint(sizeof(uint)))"
        XCTAssertTrue(msl.contains("wgpu_store_n((*p), 100000000u, \(count), 7u)"), "a write through a pointer is unclamped\n\(msl)")
        XCTAssertTrue(msl.contains("wgpu_store_n((*alsoP), idx, \(count),"), "a pointer copied from a pointer is unclamped\n\(msl)")
        XCTAssertTrue(msl.contains("wgpu_at_n((*p), idx, \(count))"), "a read through a pointer is unclamped\n\(msl)")
        XCTAssertFalse(msl.contains("wgpu_at((*"), msl)
        XCTAssertFalse(msl.contains("wgpu_store((*"), msl)
    }

    /// A storage struct spelled **through an alias** still finds its runtime-sized member — and the
    /// pipeline still binds the size table (the emitter and the pipeline must agree).
    func test_aRuntimeArrayMemberOfAnAliasedStructIsClamped() throws {
        let source = """
        struct Buf { head: u32, items: array<u32> }
        alias Inner = Buf;
        alias B = Inner;
        @group(0) @binding(0) var<storage, read_write> b: B;
        @group(0) @binding(1) var<uniform> idx: u32;
        @compute @workgroup_size(1) fn cs() {
            b.items[idx] = b.items[idx + 1u] + b.head;
        }
        """
        let msl = try translate(source, entryPoints: ["cs"])
        // items starts at offset 4; the subtraction saturates at 0.
        let count = "((wgpu_buffer_sizes[0] > 4u ? wgpu_buffer_sizes[0] - 4u : 0u) / uint(sizeof(uint)))"
        XCTAssertTrue(msl.contains("wgpu_store_n(b.items, idx, \(count),"), "a write through an alias is unclamped\n\(msl)")
        XCTAssertTrue(msl.contains("wgpu_at_n(b.items, idx + 1u, \(count))"), "a read through an alias is unclamped\n\(msl)")
        XCTAssertTrue(
            try WGSLShaderModule(source: source).usesArrayLength(entryPoints: ["cs"]),
            "the clamp reads the size table, so the pipeline must bind it"
        )
    }

    /// A pointer to the **whole struct** (`(*q).items[i]`) resolves to the same member bound.
    func test_aRuntimeArrayMemberReachedThroughAStructPointerIsClamped() throws {
        let msl = try translate("""
        struct Buf { head: u32, items: array<u32> }
        @group(0) @binding(0) var<storage, read_write> buf: Buf;
        @group(0) @binding(1) var<uniform> idx: u32;
        @compute @workgroup_size(1) fn cs() {
            let q = &buf;
            (*q).items[idx] = (*q).head;
            buf.head = arrayLength(&(*q).items);
        }
        """, entryPoints: ["cs"])
        let count = "((wgpu_buffer_sizes[0] > 4u ? wgpu_buffer_sizes[0] - 4u : 0u) / uint(sizeof(uint)))"
        XCTAssertTrue(msl.contains("wgpu_store_n((*q).items, idx, \(count),"), "a write through a struct pointer is unclamped\n\(msl)")
        XCTAssertTrue(msl.contains("buf.head = \(count)"), "arrayLength() through a pointer\n\(msl)")
    }

    /// A pointer the emitter **cannot** bound — a `ptr<storage>` parameter points wherever the caller
    /// chose — must not fall back to an unchecked access: the prelude deletes the bare pointer
    /// overloads, so the shader fails to compile instead.
    func test_aRuntimeArrayIndexWithoutABoundDoesNotCompile() throws {
        let module = try WGSLShaderModule(source: """
        struct Buf { head: u32, items: array<u32> }
        @group(0) @binding(0) var<storage, read_write> buf: Buf;
        @group(0) @binding(1) var<uniform> idx: u32;
        fn poke(q: ptr<storage, Buf, read_write>, i: u32) { (*q).items[i] = (*q).items[0]; }
        @compute @workgroup_size(1) fn cs() { poke(&buf, idx); }
        """)
        let groups = module.autoBindGroupLayouts(entryPoints: ["cs"])
        let msl = try module.translateToMSL(
            entryPoints: ["cs"], bindings: try WGSLBindingAssigner.assign(groups: groups)
        )
        XCTAssertTrue(msl.contains("wgpu_store((*q).items, i,"), msl)
        XCTAssertFalse(MSLPrelude.source.contains("p[uint(i)]"), "an unbounded pointer overload is back in the prelude")
        if let result = MetalCompilerHarness.compile(msl) {
            XCTAssertFalse(result.succeeded, "an unbounded runtime array index compiled\n\(msl)")
        }
    }

    /// A binding too small to hold one element gives a count of 0 — the store helper must then
    /// **drop** the write rather than fall back to slot 0, which may lie past the binding's end.
    func test_aRuntimeArrayStoreWithNoElementsIsDropped() {
        XCTAssertTrue(
            MSLPrelude.source.contains("wgpu_store_n(device T* p, I i, uint count, V value) { if (count != 0u) {"),
            "a write with a count of 0 is no longer dropped"
        )
    }

    /// A local that **shadows** a runtime-array global is not that global — it keeps the fixed-size clamp.
    func test_aLocalShadowingARuntimeArrayGlobalIsNotBoundByItsTable() throws {
        let msl = try translate("""
        @group(0) @binding(0) var<storage, read_write> data: array<u32>;
        @group(0) @binding(1) var<uniform> idx: u32;
        @compute @workgroup_size(1) fn cs() {
            data[0] = 1u;
            var data = array<u32, 4>();
            data[idx] = 2u;
        }
        """, entryPoints: ["cs"])
        XCTAssertTrue(msl.contains("wgpu_store_n(data, 0, "), msl)
        XCTAssertTrue(msl.contains("wgpu_store(wgpu_shadow_data, idx, 2u)"), msl)
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
}
