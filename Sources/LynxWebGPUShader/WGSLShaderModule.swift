import Foundation
import LynxWebGPUCore

/// A parsed WGSL shader module.
///
/// At `createShaderModule` time it does **parsing and reflection only.** MSL emission is deferred to
/// pipeline creation, because binding indices can only be assigned once the pipeline layout is known
/// (`translateToMSL(entryPoints:bindings:)`).
public final class WGSLShaderModule {
    /// The original WGSL.
    public let source: String
    /// Entry point and binding information needed to create a pipeline.
    public let reflection: WGSLShaderReflection

    private let ast: WGSLModule
    /// Functions needing the buffer size table — computed with the AST walks at creation, so no
    /// later query walks the tree on the caller's stack.
    private let bufferSizeUsers: Set<String>

    /// Parsing and reflection run on `WGSLDeepStack` — the source is untrusted and both recurse
    /// over its nesting.
    public init(source: String) throws {
        self.source = source
        let (ast, reflection, bufferSizeUsers) = try WGSLDeepStack.run {
            let ast = try WGSLParser.parse(source)
            return (
                ast, WGSLReflectionBuilder.build(ast),
                WGSLReflectionBuilder.functionsNeedingBufferSizes(in: ast)
            )
        }
        self.ast = ast
        self.reflection = reflection
        self.bufferSizeUsers = bufferSizeUsers
    }

    /// Builds MSL source containing the requested entry points.
    ///
    /// - Parameters:
    ///   - entryPoints: the entry point names this pipeline uses (vertex/fragment, or one compute).
    ///   - bindings: `@group/@binding` → Metal index assignment.
    ///   - constants: pipeline constant (`override`) values. They override the shader's defaults.
    public func translateToMSL(
        entryPoints: [String],
        bindings: WGSLBindingAssignment,
        constants: [String: Double] = [:]
    ) throws -> String {
        // Emission recurses over the shader's nesting — on `WGSLDeepStack`, like parsing.
        let ast = self.ast, reflection = self.reflection
        return try WGSLDeepStack.run {
            var emitter = MSLEmitter(
                module: try Self.applying(constants, to: ast), reflection: reflection, bindings: bindings
            )
            return try emitter.emit(entryPoints: entryPoints)
        }
    }

    /// Folds pipeline constants into the AST up front.
    ///
    /// To make the emitter and the layout calculator read the same constant table, **planting the
    /// values in the AST** is safer than carrying them separately (it naturally covers the case where
    /// an array length depends on an override).
    ///
    /// Each value is converted to the override's WGSL type the way the spec does it (WebIDL
    /// conversion — see `literal(_:for:type:)`); a value the type cannot hold is a validation error,
    /// not a trap.
    private static func applying(_ constants: [String: Double], to module: WGSLModule) throws -> WGSLModule {
        guard !constants.isEmpty else { return module }
        var updated = module
        let typer = WGSLOverrideTyper(module: module)
        updated.constants = try module.constants.map { constant in
            guard constant.isOverride, let supplied = constants[constant.name] else { return constant }
            var replaced = constant
            let type = typer.overrideScalarType(of: constant)
            replaced.value = try literal(supplied, for: constant.name, type: type)
            // An untyped override takes its type from its initializer (`override s = 0.5;` is f32).
            // Pin it, so a supplied `2` does not turn the declaration into an int.
            if replaced.type == nil, let type { replaced.type = .scalar(type) }
            return replaced
        }
        return updated
    }

    /// The spec's "converting an IDL value to a WGSL type" for a pipeline constant.
    ///
    /// - `bool`: IDL `boolean` (0, -0 and NaN are false).
    /// - `i32` / `u32`: `[EnforceRange] long` / `unsigned long` — non-finite or out-of-range values
    ///   are rejected, fractions are truncated toward zero.
    /// - `f32`: IDL `float` — must be finite and must not round to infinity.
    /// - `f16`: as `f32`, and |value| must not exceed 65504 (the largest finite f16). This is the
    ///   rule Dawn applies (`IsDoubleValueRepresentableAsF16`) — the same bundle must be accepted or
    ///   rejected alike on both backends.
    ///
    /// `type == nil` means the declaration's type could not be worked out syntactically, so the
    /// initializer is not provably integral; the value is then planted as an f32 (planting an int
    /// would let MSL infer `int`, and `s / 4` would silently become integer division).
    private static func literal(_ value: Double, for name: String, type: String?) throws -> WGSLExpression {
        func reject(_ reason: String) -> WGPUError {
            WGPUError.validation(
                "pipeline constant '\(name)' = \(value) cannot be converted to \(type ?? "the override's type"): \(reason)"
            )
        }
        func enforceRange(_ lower: Double, _ upper: Double) throws -> Double {
            guard value.isFinite else { throw reject("the value must be finite") }
            let truncated = value.rounded(.towardZero)
            guard truncated >= lower, truncated <= upper else {
                throw reject("out of range [\(Int64(lower)), \(Int64(upper))]")
            }
            return truncated
        }
        func float(maximumMagnitude: Double? = nil) throws -> WGSLExpression {
            guard value.isFinite else { throw reject("the value must be finite") }
            let single = Float(value)
            guard single.isFinite else { throw reject("the value overflows f32") }
            if let maximumMagnitude, abs(value) > maximumMagnitude {
                throw reject("the value is outside f16's range [-\(Int(maximumMagnitude)), \(Int(maximumMagnitude))]")
            }
            return .floatLiteral(String(single))
        }

        switch type {
        case "bool":
            return .boolLiteral(!(value == 0 || value.isNaN))
        case "i32":
            let integer = Int32(try enforceRange(-2_147_483_648, 2_147_483_647))
            if integer == .min {
                // `-2147483648` is `-(2147483648)` in C++, whose literal is not an int — spell it like the prelude.
                return .paren(.binary(op: "-", .unary(op: "-", .intLiteral("2147483647")), .intLiteral("1")))
            }
            // WGSL has no negative literals — a negative value is negation applied to one.
            if integer < 0 { return .unary(op: "-", .intLiteral(String(integer.magnitude))) }
            return .intLiteral(String(integer))
        case "u32":
            return .intLiteral(String(UInt32(try enforceRange(0, 4_294_967_295))) + "u")
        case "f32":
            return try float()
        case "f16":
            return try float(maximumMagnitude: 65504)
        default:
            return try float()
        }
    }

    /// Derives the bind group layouts a `layout: "auto"` pipeline uses from the shader declarations.
    ///
    /// Returns an array in group index order, with an empty array where a group is unused (Metal index
    /// assignment depends on group order, so the slot has to stay).
    public func autoBindGroupLayouts(entryPoints: [String]) -> [[WGPUBindGroupLayoutEntry]] {
        let used = reflection.resources(usedBy: entryPoints)
        guard let maximumGroup = used.map(\.group).max() else { return [] }

        var groups: [[WGPUBindGroupLayoutEntry]] = Array(repeating: [], count: maximumGroup + 1)
        for resource in used {
            groups[resource.group].append(WGPUBindGroupLayoutEntry(
                binding: resource.binding,
                visibility: reflection.visibility(of: resource.name),
                layout: resource.bindingLayout
            ))
        }
        return groups.map { $0.sorted { $0.binding < $1.binding } }
    }

    /// The **MSL function name** corresponding to a WGSL entry point name.
    ///
    /// Names MSL rejects, such as `main`, are renamed during emission, so this value is what must be
    /// passed to `MTLLibrary.makeFunction(name:)`.
    public static func mslFunctionName(for entryPoint: String) -> String {
        MSLTypeMapping.functionName(entryPoint)
    }

    /// Whether these entry points need the buffer size table.
    ///
    /// Needed when they use `arrayLength()` or **index a runtime-sized array** — the latter because
    /// clamping a range requires knowing the bound (robustness). When needed, the runtime must plug
    /// the table into the reserved index (`WGSLMetalLimits.bufferSizesIndex`).
    ///
    /// **This must compute the same thing as the emitter** — a mismatch makes the shader read an unbound buffer.
    public func usesArrayLength(entryPoints: [String]) -> Bool {
        entryPoints.contains(where: bufferSizeUsers.contains)
    }

    /// The entry point's `@workgroup_size` as a pipeline dispatches it — used as threadsPerThreadgroup.
    ///
    /// The arguments are override-expressions, so they are evaluated with the pipeline's `constants`
    /// applied (the same ones `translateToMSL` gets) and then checked against the device `limits`.
    /// A size that cannot be computed, or one out of bounds, is a pipeline-creation validation error.
    /// nil when the entry point does not exist or declares no workgroup size.
    public func workgroupSize(
        of entryPoint: String,
        constants: [String: Double] = [:],
        limits: WGSLWorkgroupLimits = .specDefaults
    ) throws -> (x: Int, y: Int, z: Int)? {
        let module = try Self.applying(constants, to: ast)
        guard let function = module.functionNamed(entryPoint),
              let size = try module.workgroupSize(of: function) else { return nil }

        func failure(_ reason: String) -> WGPUError {
            WGPUError.validation(
                "@workgroup_size of '\(entryPoint)' is (\(size.x), \(size.y), \(size.z)) — \(reason)"
            )
        }
        for (value, limit, name) in [(size.x, limits.maxSizeX, "X"), (size.y, limits.maxSizeY, "Y"),
                                     (size.z, limits.maxSizeZ, "Z")] where value > limit {
            throw failure("\(value) exceeds maxComputeWorkgroupSize\(name) (\(limit))")
        }
        // Each dimension is within its limit by now, so the product cannot overflow.
        let invocations = size.x * size.y * size.z
        guard invocations <= limits.maxInvocations else {
            throw failure("\(invocations) invocations exceed maxComputeInvocationsPerWorkgroup (\(limits.maxInvocations))")
        }
        return size
    }

    /// The spec's **"get the entry point"** — with no name, use the stage's **only** entry point.
    ///
    /// `entryPoint` is not required by the spec. Omitted, it uses the entry point when exactly one
    /// exists for that stage, and is an error when there is none or more than one. Guessing `"main"`
    /// would reject entire shaders whose entry points are named differently — three.js's mipmap
    /// shaders (`mainVS` + `main_2d`, …) really did break that way.
    public func resolveEntryPoint(_ requested: String?, stage: WGSLStage) throws -> String {
        if let requested {
            _ = try requireEntryPoint(requested, stage: stage)
            return requested
        }
        let candidates = reflection.entryPoints.filter { $0.stage == stage }
        guard let only = candidates.first, candidates.count == 1 else {
            let available = reflection.entryPoints.map { "\($0.name)(\($0.stage.rawValue))" }
            throw WGPUError.validation(
                candidates.isEmpty
                    ? "the shader has no \(stage.rawValue) entry point (available: \(available.joined(separator: ", ")))"
                    : "the shader has \(candidates.count) \(stage.rawValue) entry points, so one cannot be chosen — "
                        + "specify entryPoint (\(candidates.map(\.name).joined(separator: ", ")))"
            )
        }
        return only.name
    }

    /// Checks the entry point exists and belongs to the expected stage.
    public func requireEntryPoint(_ name: String, stage: WGSLStage) throws -> WGSLEntryPointInfo {
        guard let entryPoint = reflection.entryPoint(named: name) else {
            let available = reflection.entryPoints.map { "\($0.name)(\($0.stage.rawValue))" }
            throw WGPUError.validation(
                "the shader has no entry point '\(name)' (available: \(available.joined(separator: ", ")))"
            )
        }
        guard entryPoint.stage == stage else {
            throw WGPUError.validation(
                "entry point '\(name)' is a \(entryPoint.stage.rawValue) shader but was used as \(stage.rawValue)"
            )
        }
        return entryPoint
    }
}

/// Works out the WGSL type of an `override` from its declaration or, when untyped, from the
/// **syntax** of its initializer (AbstractInt → i32, AbstractFloat → f32).
///
/// The shader is user input, so every walk is bounded in both time and stack:
/// - alias walks are **iterative**, with one visited set and a hop budget (`aliasHopBudget`) — an
///   invalid `alias A = array<A, 2>;` or an absurdly long chain gives nil (the f32 fallback);
/// - constant references are memoized per name (`const x1 = x0 + x0; const x2 = x1 + x1; …` stays
///   linear, not 2^N), and a constant that refers back to itself is cut off;
/// - expression recursion stops at `expressionDepthLimit`, and a result cut short that way is never
///   memoized (it would poison later lookups of the same constant).
private final class WGSLOverrideTyper {
    private let module: WGSLModule
    /// Alias name → target (the first declaration wins, like `first(where:)`). Built once.
    private let aliases: [String: WGSLType]
    /// Module constant name → declaration. Built once.
    private let constants: [String: WGSLModuleConstant]
    /// Constant name → its component type (nil = could not be typed). Filled as the walk goes.
    private var constantTypes: [String: String?] = [:]
    private var constantsInProgress: Set<String> = []
    /// Set when the depth guard trips during the current constant's walk — its result is then partial.
    private var depthCutOff = false

    private static let aliasHopBudget = 64
    private static let expressionDepthLimit = 256

    init(module: WGSLModule) {
        self.module = module
        self.aliases = Dictionary(module.aliases.map { ($0.name, $0.type) }, uniquingKeysWith: { first, _ in first })
        self.constants = Dictionary(module.constants.map { ($0.name, $0) }, uniquingKeysWith: { first, _ in first })
    }

    /// The scalar type name of an `override`. nil when the initializer is too complex to read.
    func overrideScalarType(of constant: WGSLModuleConstant) -> String? {
        depthCutOff = false
        if let type = constant.type {
            guard case .scalar(let name)? = resolvingAliases(type) else { return nil }
            return Self.scalarNames.contains(name) ? name : nil
        }
        guard let value = constant.value else { return nil }
        return componentType(of: value, depth: 0)
    }

    private static let scalarNames: Set<String> = ["bool", "i32", "u32", "f32", "f16"]

    /// Follows an alias chain (`alias Count2 = Count; alias Count = u32;`). nil on a cycle or past
    /// the hop budget.
    private func resolvingAliases(_ type: WGSLType) -> WGSLType? {
        var resolved = type
        var visited: Set<String> = []
        while case .named(let name) = resolved, let target = aliases[name] {
            guard visited.count < Self.aliasHopBudget, visited.insert(name).inserted else { return nil }
            resolved = target
        }
        return resolved
    }

    /// The scalar a type is made of (`vec3<u32>` → u32, `array<f32, 4>` → f32). nil for structs, and
    /// for a type that refers back to itself through aliases (`alias A = vec2<A>;`) or whose alias
    /// chain exceeds the hop budget. Iterative — element nesting without aliases is a finite tree.
    private func componentName(of type: WGSLType) -> String? {
        var current = type
        var visited: Set<String> = []
        while true {
            switch current {
            case .scalar(let name):
                return Self.scalarNames.contains(name) ? name : nil
            case .named(let name):
                guard visited.count < Self.aliasHopBudget, visited.insert(name).inserted,
                      let target = aliases[name] else { return nil }
                current = target
            case .vector(_, let element), .matrix(_, _, let element), .array(let element, _), .atomic(let element):
                current = element
            default:
                return nil
            }
        }
    }

    /// Builtins whose result has the (unified) component type of their arguments, integer or float.
    private static let sameTypeBuiltins: Set<String> = ["min", "max", "abs", "clamp", "sign", "dot"]

    /// Float-only builtins whose result has the component type of their arguments.
    private static let floatBuiltins: Set<String> = [
        "pow", "sqrt", "inverseSqrt", "exp", "exp2", "log", "log2", "floor", "ceil", "round", "trunc",
        "fract", "sin", "cos", "tan", "asin", "acos", "atan", "atan2", "sinh", "cosh", "tanh",
        "radians", "degrees", "mix", "step", "smoothstep", "fma", "saturate",
        "length", "distance", "determinant", "normalize",
    ]

    /// Integer bit builtins — the result has the type of the first argument.
    private static let bitBuiltins: Set<String> = [
        "countOneBits", "countLeadingZeros", "countTrailingZeros", "reverseBits",
        "firstLeadingBit", "firstTrailingBit", "extractBits", "insertBits",
    ]

    /// Builtins with a fixed result type.
    private static let fixedTypeBuiltins: [String: String] = [
        "all": "bool", "any": "bool",
        "pack4x8snorm": "u32", "pack4x8unorm": "u32", "pack2x16snorm": "u32", "pack2x16unorm": "u32",
        "pack2x16float": "u32", "pack4xI8": "u32", "pack4xU8": "u32",
        "dot4U8Packed": "u32", "dot4I8Packed": "i32",
    ]

    /// `mat2x2` … `mat4x4` — exactly (a struct named `matrix` is not one).
    private static func isMatrixConstructor(_ name: String) -> Bool {
        let characters = Array(name)
        return characters.count == 6 && name.hasPrefix("mat") && characters[4] == "x"
            && ("2"..."4").contains(characters[3]) && ("2"..."4").contains(characters[5])
    }

    /// The common type of operands that must agree — mixed operands only type-check when one side is
    /// abstract, so the concrete/float side wins. nil when any operand is unknown or the mix is invalid.
    private func unifiedType(of operands: [WGSLExpression], depth: Int) -> String? {
        var types: Set<String> = []
        for operand in operands {
            guard let type = componentType(of: operand, depth: depth) else { return nil }
            types.insert(type)
        }
        if types.count <= 1 { return types.first }
        if types.contains("bool") { return nil }
        if types.contains("f16") { return "f16" }
        if types.contains("f32") { return "f32" }
        return "u32"
    }

    /// The type a module constant evaluates to — computed once per name.
    private func constantType(named name: String, depth: Int) -> String? {
        if let cached = constantTypes[name] { return cached }
        guard !constantsInProgress.contains(name), let constant = constants[name] else { return nil }
        constantsInProgress.insert(name)
        let outerCutOff = depthCutOff
        depthCutOff = false
        defer {
            constantsInProgress.remove(name)
            // A cut-short walk taints everything that contains it, not just this constant.
            depthCutOff = depthCutOff || outerCutOff
        }
        let type: String?
        if let declared = constant.type {
            type = componentName(of: declared)
        } else if let value = constant.value {
            type = componentType(of: value, depth: depth + 1)
        } else {
            type = nil
        }
        if !depthCutOff { constantTypes[name] = .some(type) }
        return type
    }

    /// The scalar type of an expression, or the **component** type of a vector/array expression
    /// (`vec2i(1, 2)` → i32, so `.x` / `[0]` on it is i32 too). An override is always scalar, so the
    /// distinction only matters inside — `dot(a, b)` and `length(v)` take their type from components.
    private func componentType(of expression: WGSLExpression, depth: Int) -> String? {
        // Expression trees are finite and constant references are memoized and cycle-checked; this
        // only bounds the stack for absurdly deep nesting.
        guard depth < Self.expressionDepthLimit else {
            depthCutOff = true
            return nil
        }
        switch expression {
        case .boolLiteral:
            return "bool"
        case .intLiteral(let text):
            return text.hasSuffix("u") ? "u32" : "i32"
        case .floatLiteral(let text):
            return text.hasSuffix("h") ? "f16" : "f32"
        case .paren(let inner), .member(let inner, _), .index(let inner, _):
            return componentType(of: inner, depth: depth + 1)
        case .unary(let op, let inner):
            return op == "!" ? "bool" : componentType(of: inner, depth: depth + 1)
        case .binary(let op, let left, let right):
            if ["==", "!=", "<", "<=", ">", ">=", "&&", "||"].contains(op) { return "bool" }
            // A shift takes the left operand's type (`1 << 2u` is i32).
            if op == "<<" || op == ">>" { return componentType(of: left, depth: depth + 1) }
            return unifiedType(of: [left, right], depth: depth + 1)
        case .identifier(let name):
            return constantType(named: name, depth: depth)
        case .call(let callee, let typeArguments, let arguments):
            // A struct constructor — its members are not read here (it may well be named `matrix`).
            if module.structNamed(callee) != nil { return nil }
            if aliases[callee] != nil { return componentName(of: .named(callee)) }
            if Self.scalarNames.contains(callee) { return callee }
            if callee == "bitcast" {
                return typeArguments.first.flatMap { componentName(of: $0) }
            }
            // Constructors: `vec2<i32>(…)`, `vec2i(…)`, `array<u32, 2>(…)`, `vec2(1, 2)` (from the arguments).
            if let shorthand = WGSLParser.shorthandType(callee) { return componentName(of: shorthand) }
            if ["array", "vec2", "vec3", "vec4"].contains(callee) || Self.isMatrixConstructor(callee) {
                if let element = typeArguments.first { return componentName(of: element) }
                return unifiedType(of: arguments, depth: depth + 1)
            }
            if let fixed = Self.fixedTypeBuiltins[callee] { return fixed }
            if callee == "select", arguments.count == 3 {
                // select(f, t, cond) — the value operands decide the type.
                return unifiedType(of: Array(arguments.prefix(2)), depth: depth + 1)
            }
            if Self.bitBuiltins.contains(callee), let first = arguments.first {
                return componentType(of: first, depth: depth + 1)
            }
            if Self.sameTypeBuiltins.contains(callee) {
                return unifiedType(of: arguments, depth: depth + 1)
            }
            if Self.floatBuiltins.contains(callee) {
                // An AbstractInt argument converts to AbstractFloat here (`sqrt(4)` is f32).
                let type = unifiedType(of: arguments, depth: depth + 1)
                return type == "i32" ? "f32" : type
            }
            return nil
        default:
            return nil
        }
    }
}
