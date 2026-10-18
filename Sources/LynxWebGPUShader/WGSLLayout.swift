import Foundation
import LynxWebGPUCore

/// Memory layout calculator for WGSL host-shared (uniform/storage) types.
///
/// **Why it is needed** — WGSL and MSL layout rules are nearly identical but diverge on `vec3`.
/// WGSL `vec3<f32>` has alignment 16 and size 12, so `struct { pos: vec3f, r: f32 }` is 16 bytes;
/// MSL `float3` has size 16, making the same struct 32 bytes. For JS to fill a uniform buffer by
/// WGSL rules and have it read back correctly, **the MSL struct must be padded to WGSL offsets**.
/// The emitter (`MSLEmitter`) uses this calculation to choose `packed_float3` and insert padding.
///
/// Reference: the WGSL spec's "Memory Layout" (AlignOf/SizeOf/RoundUp).
enum WGSLLayout {
    struct MemberPlacement {
        var name: String
        var type: WGSLType
        var offset: Int
        var size: Int
        var align: Int
        /// A vec3 whose next member starts within offset+16, so `float3` (16 bytes) cannot express it.
        var needsPackedVector: Bool
    }

    struct StructPlacement {
        var members: [MemberPlacement]
        var size: Int
        var align: Int
    }

    static func roundUp(_ alignment: Int, _ value: Int) -> Int {
        guard alignment > 0 else { return value }
        return ((value + alignment - 1) / alignment) * alignment
    }

    // MARK: - Alignment / size

    static func align(of type: WGSLType, module: WGSLModule, uniform: Bool) -> Int {
        switch type {
        case .scalar(let name):
            return name == "f16" ? 2 : 4
        case .vector(let size, let element):
            let elementAlign = align(of: element, module: module, uniform: uniform)
            return (size == 2 ? 2 : 4) * elementAlign
        case .matrix(_, let rows, let element):
            return align(of: .vector(size: rows, element: element), module: module, uniform: uniform)
        case .array(let element, _):
            let elementAlign = align(of: element, module: module, uniform: uniform)
            return uniform ? roundUp(16, elementAlign) : elementAlign
        case .atomic:
            return 4
        case .named(let name):
            guard let structure = resolveStruct(name, module: module) else { return 4 }
            let placement = layout(of: structure, module: module, uniform: uniform)
            return placement.align
        case .pointer, .texture, .sampler, .void:
            return 4
        }
    }

    static func size(of type: WGSLType, module: WGSLModule, uniform: Bool) -> Int {
        switch type {
        case .scalar(let name):
            return name == "f16" ? 2 : 4
        case .vector(let count, let element):
            return count * size(of: element, module: module, uniform: uniform)
        case .matrix(let columns, let rows, let element):
            let column = WGSLType.vector(size: rows, element: element)
            let stride = roundUp(
                align(of: column, module: module, uniform: uniform),
                size(of: column, module: module, uniform: uniform)
            )
            return columns * stride
        case .array(let element, let count):
            let stride = arrayStride(element: element, module: module, uniform: uniform)
            guard let count, let literal = constantValue(count, module: module) else { return stride }
            return literal * stride
        case .atomic:
            return 4
        case .named(let name):
            guard let structure = resolveStruct(name, module: module) else { return 4 }
            return layout(of: structure, module: module, uniform: uniform).size
        case .pointer, .texture, .sampler, .void:
            return 4
        }
    }

    static func arrayStride(element: WGSLType, module: WGSLModule, uniform: Bool) -> Int {
        let elementAlign = align(of: element, module: module, uniform: uniform)
        let stride = roundUp(elementAlign, size(of: element, module: module, uniform: uniform))
        return uniform ? roundUp(16, stride) : stride
    }

    // MARK: - Struct layout

    static func layout(of structure: WGSLStruct, module: WGSLModule, uniform: Bool) -> StructPlacement {
        var members: [MemberPlacement] = []
        var offset = 0
        var maximumAlign = 1

        for member in structure.members {
            var memberAlign = align(of: member.type, module: module, uniform: uniform)
            var memberSize = size(of: member.type, module: module, uniform: uniform)
            if let explicit = member.attributes.first(named: "align")?.integerValue { memberAlign = explicit }
            if let explicit = member.attributes.first(named: "size")?.integerValue { memberSize = max(memberSize, explicit) }

            offset = roundUp(memberAlign, offset)
            members.append(MemberPlacement(
                name: member.name, type: member.type,
                offset: offset, size: memberSize, align: memberAlign,
                needsPackedVector: false
            ))
            offset += memberSize
            maximumAlign = max(maximumAlign, memberAlign)
        }

        let structAlign = uniform ? roundUp(16, maximumAlign) : maximumAlign
        // If a member after a vec3 uses bytes 12...15, MSL `float3` (16 bytes) cannot line it up.
        for index in members.indices {
            guard case .vector(3, _) = members[index].type else { continue }
            let end = members[index].offset + 16
            let nextOffset = index + 1 < members.count
                ? members[index + 1].offset
                : roundUp(structAlign, offset)
            members[index].needsPackedVector = nextOffset < end
        }

        return StructPlacement(members: members, size: roundUp(structAlign, offset), align: structAlign)
    }

    // MARK: - Helpers

    static func resolveStruct(_ name: String, module: WGSLModule) -> WGSLStruct? {
        // Follow aliases, including chains (`alias A = B; alias B = Buf;`). WGSL forbids cycles;
        // the visited set only keeps a malformed module from looping.
        var name = name
        var visited = Set<String>()
        while visited.insert(name).inserted {
            if let structure = module.structNamed(name) { return structure }
            guard let alias = module.aliases.first(where: { $0.name == name }),
                  case .named(let target) = alias.type else { return nil }
            name = target
        }
        return nil
    }

    /// Computes a value that must be known at compile time, such as an array length.
    ///
    /// An array length is an override-expression and may sit inside a function, so overrides and
    /// function-local `const`s resolve too. nil when it cannot be computed (`WGSLConstantEvaluator`
    /// says why).
    static func constantValue(_ expression: WGSLExpression, module: WGSLModule) -> Int? {
        try? arrayLength(expression, module: module)
    }

    /// `constantValue` that keeps the reason it failed.
    static func arrayLength(_ expression: WGSLExpression, module: WGSLModule) throws -> Int {
        var evaluator = WGSLConstantEvaluator(module: module, allowsOverrides: true, allowsFunctionConstants: true)
        return try evaluator.evaluate(expression)
    }

    /// Names of structs used in the uniform address space — their layout rules are stricter than storage.
    static func uniformStructNames(_ module: WGSLModule) -> Set<String> {
        var names = Set<String>()
        for global in module.globals where global.addressSpace == "uniform" {
            markStructs(in: global.type, module: module, into: &names)
        }
        return names
    }

    private static func markStructs(in type: WGSLType, module: WGSLModule, into names: inout Set<String>) {
        switch type {
        case .named(let name):
            guard !names.contains(name), let structure = resolveStruct(name, module: module) else { return }
            names.insert(name)
            for member in structure.members {
                markStructs(in: member.type, module: module, into: &names)
            }
        case .array(let element, _), .atomic(let element):
            markStructs(in: element, module: module, into: &names)
        case .pointer(_, let element, _):
            markStructs(in: element, module: module, into: &names)
        default:
            break
        }
    }
}
