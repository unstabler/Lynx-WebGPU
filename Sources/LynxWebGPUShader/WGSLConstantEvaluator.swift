import Foundation
import LynxWebGPUCore

/// A WGSL integer literal, read by the spec grammar (`decimal_int_literal` / `hex_int_literal`).
///
/// **The one place integer literal text becomes a number.** Stripping non-digits turned
/// `0x10` into 10 and `1+1` into 11 — every compile-time integer (attribute arguments, array
/// lengths) goes through here instead.
struct WGSLIntegerLiteral: Equatable {
    enum Suffix: Equatable {
        /// No suffix — an AbstractInt (i64 range).
        case none
        /// `i` — i32.
        case i
        /// `u` — u32.
        case u
    }

    var value: Int
    var suffix: Suffix

    /// nil when the text is not a well-formed literal or its value does not fit its type.
    init?(_ text: String) {
        var body = Substring(text)
        switch body.last {
        case "i": suffix = .i; body = body.dropLast()
        case "u": suffix = .u; body = body.dropLast()
        default: suffix = .none
        }

        let parsed: Int?
        if body.hasPrefix("0x") || body.hasPrefix("0X") {
            let digits = body.dropFirst(2)
            guard !digits.isEmpty, digits.allSatisfy(Self.isASCIIHexDigit) else { return nil }
            parsed = Int(digits, radix: 16)
        } else {
            // `0` alone, or no leading zero (`08` is not WGSL). ASCII only — `isNumber` lets other scripts' digits through.
            guard let first = body.first, body.allSatisfy(Self.isASCIIDigit),
                  first != "0" || body.count == 1 else { return nil }
            parsed = Int(body)
        }
        // `Int` is 64-bit on every supported platform, so a failed parse is AbstractInt overflow.
        guard let parsed else { return nil }

        switch suffix {
        case .none: break
        case .i: guard parsed <= Int(Int32.max) else { return nil }
        case .u: guard parsed <= Int(UInt32.max) else { return nil }
        }
        value = parsed
    }

    private static func isASCIIDigit(_ character: Character) -> Bool {
        guard let ascii = character.asciiValue else { return false }
        return ascii >= 0x30 && ascii <= 0x39
    }

    private static func isASCIIHexDigit(_ character: Character) -> Bool {
        guard let ascii = character.asciiValue else { return false }
        return (ascii >= 0x30 && ascii <= 0x39) || (ascii >= 0x41 && ascii <= 0x46) || (ascii >= 0x61 && ascii <= 0x66)
    }
}

/// Evaluates an integer const-expression (or override-expression) at compile time.
///
/// Covers integer literals, names of `const`s (and, where allowed, `override`s), parentheses,
/// unary `-`, `+ - * / %` and the conversions `i32(e)` / `u32(e)`. Anything else **throws with a reason** rather than guessing — a
/// wrong number here becomes a wrong binding slot or a wrong struct layout, both silent.
struct WGSLConstantEvaluator {
    /// Why a value could not be computed. Callers wrap it into a `WGPUError` with their own context.
    struct Failure: Error, Equatable {
        var reason: String
        /// The expression names an `override` where overrides are not allowed. An override-expression
        /// position (`@workgroup_size`) evaluated before pipeline constants are known defers on this.
        var referencesOverride = false
    }

    let module: WGSLModule
    /// Whether `override` names resolve (to their default or supplied value). Attributes such as
    /// `@binding` take a **const**-expression, so they must not; array lengths and
    /// `@workgroup_size` take an override-expression, so they may.
    var allowsOverrides: Bool
    /// Whether a name may resolve to a function-local `const` (array lengths inside a function body).
    var allowsFunctionConstants: Bool

    /// Names currently being resolved — `const A = B; const B = A;` must stop, not recurse forever.
    private var resolving: Set<String> = []
    /// Current recursion depth — operator nesting and const-name chains both count. Shader text is
    /// untrusted and the JS thread's stack is small, so `1+1+…` with thousands of terms must fail
    /// cleanly instead of overflowing the stack.
    private var depth = 0
    static let maximumDepth = 256

    init(module: WGSLModule, allowsOverrides: Bool, allowsFunctionConstants: Bool) {
        self.module = module
        self.allowsOverrides = allowsOverrides
        self.allowsFunctionConstants = allowsFunctionConstants
    }

    mutating func evaluate(_ expression: WGSLExpression) throws -> Int {
        depth += 1
        defer { depth -= 1 }
        guard depth <= Self.maximumDepth else { throw Failure(reason: "the expression is nested too deeply") }

        switch expression {
        case .intLiteral(let text):
            guard let literal = WGSLIntegerLiteral(text) else {
                throw Failure(reason: "'\(text)' is not a valid integer literal (or does not fit its type)")
            }
            return literal.value
        case .paren(let inner):
            return try evaluate(inner)
        case .unary("-", let operand):
            let (value, overflow) = Int(0).subtractingReportingOverflow(try evaluate(operand))
            guard !overflow else { throw Failure(reason: "the value overflows") }
            return value
        case .binary(let op, let left, let right):
            let l = try evaluate(left)
            let r = try evaluate(right)
            let result: (partialValue: Int, overflow: Bool)
            switch op {
            case "+": result = l.addingReportingOverflow(r)
            case "-": result = l.subtractingReportingOverflow(r)
            case "*": result = l.multipliedReportingOverflow(by: r)
            case "/", "%":
                guard r != 0 else { throw Failure(reason: "division by zero") }
                result = op == "/" ? l.dividedReportingOverflow(by: r) : l.remainderReportingOverflow(dividingBy: r)
            default:
                throw Failure(reason: "operator '\(op)' is not supported in a compile-time integer")
            }
            guard !result.overflow else { throw Failure(reason: "the value overflows") }
            return result.partialValue
        case .identifier(let name):
            return try resolve(name)
        case .call(let callee, let typeArguments, let arguments)
            where (callee == "i32" || callee == "u32") && typeArguments.isEmpty && arguments.count == 1:
            // A value conversion of an integer is the identity — as long as the value fits the target.
            let value = try evaluate(arguments[0])
            let range = callee == "u32" ? 0...Int(UInt32.max) : Int(Int32.min)...Int(Int32.max)
            guard range.contains(value) else { throw Failure(reason: "\(value) does not fit in \(callee)") }
            return value
        case .floatLiteral(let text):
            throw Failure(reason: "'\(text)' is not an integer")
        default:
            throw Failure(reason: "only integer literals, const names, + - * / % and i32()/u32() are supported here")
        }
    }

    private mutating func resolve(_ name: String) throws -> Int {
        guard resolving.insert(name).inserted else {
            throw Failure(reason: "'\(name)' is defined in terms of itself")
        }
        defer { resolving.remove(name) }

        if let constant = module.constants.first(where: { $0.name == name }) {
            if constant.isOverride, !allowsOverrides {
                throw Failure(
                    reason: "'\(name)' is an override — this position needs a const-expression",
                    referencesOverride: true
                )
            }
            guard let value = constant.value else {
                throw Failure(reason: "override '\(name)' has no value (no default and none supplied)")
            }
            return try evaluate(value)
        }
        guard allowsFunctionConstants else {
            throw Failure(reason: "'\(name)' is not a module-scope const")
        }
        // A `const` declared inside a function is a compile-time constant too (`var a: array<T, maxLayers>;`).
        return try uniqueLocalConstant(named: name)
    }

    /// The value of a function-local `const` declared under this name anywhere in the module.
    ///
    /// Being local, it is visible only inside its function in principle, but there is no function
    /// context when settling an array size. So it is used **only when the module settles on one
    /// value** — if the same name holds different values per function we cannot tell which, so we
    /// give up.
    private mutating func uniqueLocalConstant(named name: String) throws -> Int {
        var found: Int?
        for function in module.functions {
            for expression in Self.localConstants(named: name, in: function.body) {
                let value = try evaluate(expression)
                if let found, found != value {
                    throw Failure(reason: "function-local consts named '\(name)' disagree (\(found) vs \(value))")
                }
                found = value
            }
        }
        guard let found else { throw Failure(reason: "'\(name)' is not a known constant") }
        return found
    }

    private static func localConstants(named name: String, in statements: [WGSLStatement]) -> [WGSLExpression] {
        var values: [WGSLExpression] = []
        for statement in statements {
            if case .constDeclaration(let declared, _, let value) = statement, declared == name {
                values.append(value)
            }
            for block in nestedBlocks(of: statement) {
                values.append(contentsOf: localConstants(named: name, in: block))
            }
        }
        return values
    }

    /// Walks nested blocks only (so a local `const` is found wherever it sits).
    private static func nestedBlocks(of statement: WGSLStatement) -> [[WGSLStatement]] {
        switch statement {
        case .ifStatement(_, let then, let elseBranch):
            switch elseBranch {
            case .block(let statements)?: return [then, statements]
            case .chained(let nested)?: return [then, [nested]]
            case nil: return [then]
            }
        case .forStatement(let initializer, _, _, let body):
            return initializer.map { [[$0], body] } ?? [body]
        case .whileStatement(_, let body):
            return [body]
        case .loopStatement(let body, let continuing):
            return continuing.map { [body, $0] } ?? [body]
        case .switchStatement(_, let cases):
            return cases.map(\.body)
        case .block(let statements):
            return [statements]
        default:
            return []
        }
    }
}

// MARK: - Attribute arguments

extension WGSLModule {
    /// The range the spec allows for an attribute whose single argument is an integer
    /// const-expression — nil for attributes that take something else.
    private static func integerRequirement(of name: String) -> (valid: (Int) -> Bool, text: String)? {
        switch name {
        case "group", "binding", "location": return ({ $0 >= 0 }, "must be non-negative")
        case "align": return ({ $0 > 0 && $0 & ($0 - 1) == 0 }, "must be a positive power of two")
        case "size": return ({ $0 > 0 }, "must be positive")
        default: return nil
        }
    }

    /// Settles `@group` `@binding` `@location` `@align` `@size` into `WGSLAttribute.integerValue`.
    ///
    /// Runs once the whole module is parsed — a module-scope `const` may be declared after its use.
    /// A value that cannot be computed is a shader-creation error: before this, an unreadable
    /// `@binding` silently dropped the resource and a hex one landed on the wrong slot.
    mutating func resolveIntegerAttributes() throws {
        let snapshot = self
        func resolve(_ attributes: inout [WGSLAttribute]) throws {
            for index in attributes.indices {
                attributes[index].integerValue = try Self.integerValue(of: attributes[index], module: snapshot)
            }
        }
        for s in structs.indices {
            for m in structs[s].members.indices { try resolve(&structs[s].members[m].attributes) }
        }
        for g in globals.indices { try resolve(&globals[g].attributes) }
        for f in functions.indices {
            try resolve(&functions[f].attributes)
            try resolve(&functions[f].returnAttributes)
            for p in functions[f].parameters.indices { try resolve(&functions[f].parameters[p].attributes) }
        }
    }

    /// A validation error quoting the attribute as written, with its line and the reason.
    static func attributeError(_ attribute: WGSLAttribute, _ reason: String) -> WGPUError {
        let written = "@\(attribute.name)(\(attribute.arguments.joined(separator: ", ")))"
        return WGPUError(
            kind: .validation,
            message: "WGSL attribute error (line \(attribute.line)): \(written) — \(reason)",
            line: attribute.line
        )
    }

    private static func integerValue(of attribute: WGSLAttribute, module: WGSLModule) throws -> Int? {
        guard let rule = integerRequirement(of: attribute.name) else { return nil }
        func failure(_ reason: String) -> WGPUError { attributeError(attribute, reason) }

        guard attribute.arguments.count == 1 else { throw failure("takes exactly one argument") }
        guard let expression = attribute.expressions.first ?? nil else {
            throw failure("'\(attribute.arguments[0])' is not an expression")
        }
        var evaluator = WGSLConstantEvaluator(module: module, allowsOverrides: false, allowsFunctionConstants: false)
        let value: Int
        do {
            value = try evaluator.evaluate(expression)
        } catch let error as WGSLConstantEvaluator.Failure {
            throw failure(error.reason)
        }
        // The spec types these as i32 or u32.
        guard value <= Int(UInt32.max) else { throw failure("\(value) does not fit in u32") }
        guard rule.valid(value) else { throw failure("\(value) \(rule.text)") }
        return value
    }
}

// MARK: - @workgroup_size

/// The device limits a compute entry point's `@workgroup_size` is checked against at pipeline
/// creation (`maxComputeWorkgroupSizeX/Y/Z`, `maxComputeInvocationsPerWorkgroup`).
public struct WGSLWorkgroupLimits: Equatable, Sendable {
    public var maxSizeX: Int
    public var maxSizeY: Int
    public var maxSizeZ: Int
    public var maxInvocations: Int

    public init(maxSizeX: Int, maxSizeY: Int, maxSizeZ: Int, maxInvocations: Int) {
        self.maxSizeX = maxSizeX
        self.maxSizeY = maxSizeY
        self.maxSizeZ = maxSizeZ
        self.maxInvocations = maxInvocations
    }

    /// The spec's default limits — what every device guarantees.
    public static let specDefaults = WGSLWorkgroupLimits(maxSizeX: 256, maxSizeY: 256, maxSizeZ: 64, maxInvocations: 256)
}

extension WGSLModule {
    /// Shader-creation checks on `@workgroup_size` — and a compute entry point must have one.
    ///
    /// The arguments are override-expressions: one that names an `override` is settled at pipeline
    /// creation (`workgroupSize(of:)` on the module the constants were applied to), a const one must
    /// already be at least 1. Each argument used to be read as `Int(text) ?? 1`, so `64u`, `0x40` or a
    /// `const` name silently dispatched one thread per workgroup.
    func validateWorkgroupSizes() throws {
        for function in functions {
            guard let attribute = function.attributes.first(named: "workgroup_size") else {
                if function.stage == .compute, let compute = function.attributes.first(named: "compute") {
                    throw Self.attributeError(
                        compute, "compute entry point '\(function.name)' has no @workgroup_size"
                    )
                }
                continue
            }
            _ = try workgroupDimensions(of: attribute, allowsOverrides: false)
        }
    }

    /// An entry point's `@workgroup_size`, evaluated — nil when it has none. Missing y / z are 1.
    ///
    /// Overrides resolve to their values **in this module**, so call it on the module the pipeline's
    /// `constants` were applied to. The dispatch size and the emitter's `var<workgroup>` zero-init
    /// stride both come from here — if they disagreed, slots past the real thread count would never
    /// be zeroed.
    func workgroupSize(of function: WGSLFunction) throws -> (x: Int, y: Int, z: Int)? {
        guard let attribute = function.attributes.first(named: "workgroup_size") else { return nil }
        // With overrides allowed nothing is deferred, so every dimension has a value.
        let values = try workgroupDimensions(of: attribute, allowsOverrides: true).map { $0 ?? 1 }
        return (values[0], values.count > 1 ? values[1] : 1, values.count > 2 ? values[2] : 1)
    }

    /// Each argument's value — nil for one that names an override while overrides are not allowed yet.
    private func workgroupDimensions(of attribute: WGSLAttribute, allowsOverrides: Bool) throws -> [Int?] {
        func failure(_ reason: String) -> WGPUError { Self.attributeError(attribute, reason) }
        guard (1...3).contains(attribute.arguments.count) else { throw failure("takes one to three arguments") }

        var values: [Int?] = []
        for index in attribute.arguments.indices {
            guard index < attribute.expressions.count, let expression = attribute.expressions[index] else {
                throw failure("'\(attribute.arguments[index])' is not an expression")
            }
            var evaluator = WGSLConstantEvaluator(
                module: self, allowsOverrides: allowsOverrides, allowsFunctionConstants: false
            )
            let value: Int
            do {
                value = try evaluator.evaluate(expression)
            } catch let error as WGSLConstantEvaluator.Failure {
                if error.referencesOverride, !allowsOverrides {
                    values.append(nil)
                    continue
                }
                throw failure(error.reason)
            }
            guard value >= 1 else { throw failure("\(value) — each dimension must be at least 1") }
            values.append(value)
        }
        return values
    }
}
