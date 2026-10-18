import Foundation

/// Type-safe reader over the `[String: Any]` a Lynx NativeModule handed us.
///
/// It never goes through a JSON string — Lynx already converts to `NSDictionary`/`NSNumber`/
/// `NSString`, and serializing once more would only add per-frame cost. Instead a failure throws
/// an **error carrying a path**, so JS learns immediately where the command stream went wrong.
///
/// **Run a host payload through `WGPUPayload.materialize(_:)` first.** This reader keeps the dictionary
/// it is given, and an `NSDictionary` received as `[String: Any]` is not a copy — only its top level is
/// native, and nested containers still point at host-owned objects for as long as the reader lives
/// (see `WGPUPayload`).
public struct WGPUValueReader {
    public let path: String
    private let dictionary: [String: Any]

    public init(_ dictionary: [String: Any], path: String = "") {
        self.dictionary = dictionary
        self.path = path
    }

    public var keys: [String] { Array(dictionary.keys) }

    public func has(_ key: String) -> Bool {
        guard let value = dictionary[key] else { return false }
        return !(value is NSNull)
    }

    private func childPath(_ key: String) -> String {
        path.isEmpty ? key : "\(path).\(key)"
    }

    /// Path of one field under this reader (`commands[3].indirectOffset`).
    ///
    /// Used when validation belongs to **the side that knows the usage**, not the side reading the
    /// value — unlike a type mismatch, a rule like "must be a multiple of 4" is known only to the caller.
    public func fieldPath(_ key: String) -> String { childPath(key) }

    private func value(_ key: String) -> Any? {
        guard let value = dictionary[key], !(value is NSNull) else { return nil }
        return value
    }

    private func missing(_ key: String) -> WGPUError {
        .validation("required field missing", path: childPath(key))
    }

    private func mismatch(_ key: String, expected: String) -> WGPUError {
        .validation("\(expected) required", path: childPath(key))
    }

    // MARK: - Integers

    /// Largest integer a JS number holds exactly (`Number.MAX_SAFE_INTEGER`, 2^53 − 1).
    public static let maxSafeInteger = 9_007_199_254_740_991

    /// The **single** conversion from a payload number to `Int` — every integer accessor goes
    /// through it.
    ///
    /// JS numbers are doubles, so NaN, ±Infinity and 1e20 all arrive here, and `Int(Double)` traps
    /// on each of them (process death — the `WGPUError` contract forbids that), while
    /// `NSNumber.intValue` silently saturates or wraps them. This follows WebIDL `[EnforceRange]`
    /// instead: non-finite → error, otherwise truncate toward zero, then outside
    /// ±`maxSafeInteger` → error. The per-type range (`unsigned long`, …) stays with the caller
    /// that knows the usage; this only guarantees the value is a real integer `Int` can hold.
    static func enforceRange(_ number: NSNumber, path: String) throws -> Int {
        let double = number.doubleValue
        guard double.isFinite else {
            throw WGPUError.validation("an integer is required (got \(double))", path: path)
        }
        let truncated = double.rounded(.towardZero)
        guard abs(truncated) <= Double(maxSafeInteger) else {
            throw WGPUError.validation(
                "\(double) is outside the safe integer range ±(2^53 - 1)", path: path)
        }
        return Int(truncated)
    }

    public func requiredInt(_ key: String) throws -> Int {
        guard let raw = value(key) else { throw missing(key) }
        guard let number = raw as? NSNumber else { throw mismatch(key, expected: "an integer") }
        return try Self.enforceRange(number, path: childPath(key))
    }

    /// A non-number value falls back like an absent one; a number that is not a usable integer throws.
    public func int(_ key: String, default fallback: Int) throws -> Int {
        try optionalInt(key) ?? fallback
    }

    /// A non-number value reads as nil like an absent one; a number that is not a usable integer throws.
    public func optionalInt(_ key: String) throws -> Int? {
        guard let number = value(key) as? NSNumber else { return nil }
        return try Self.enforceRange(number, path: childPath(key))
    }

    // MARK: - Scalars

    public func requiredDouble(_ key: String) throws -> Double {
        guard let raw = value(key) else { throw missing(key) }
        guard let number = raw as? NSNumber else { throw mismatch(key, expected: "a number") }
        return number.doubleValue
    }

    public func double(_ key: String, default fallback: Double) -> Double {
        (value(key) as? NSNumber)?.doubleValue ?? fallback
    }

    public func bool(_ key: String, default fallback: Bool) -> Bool {
        (value(key) as? NSNumber)?.boolValue ?? fallback
    }

    public func requiredString(_ key: String) throws -> String {
        guard let raw = value(key) else { throw missing(key) }
        guard let string = raw as? String else { throw mismatch(key, expected: "a string") }
        return string
    }

    public func string(_ key: String, default fallback: String) -> String {
        (value(key) as? String) ?? fallback
    }

    public func optionalString(_ key: String) -> String? {
        value(key) as? String
    }

    // MARK: - Handles

    public func requiredHandle(_ key: String) throws -> WGPUHandle {
        WGPUHandle(try requiredInt(key))
    }

    public func optionalHandle(_ key: String) throws -> WGPUHandle? {
        try optionalInt(key).map(WGPUHandle.init)
    }

    // MARK: - Enums / flags

    public func requiredEnum<T: RawRepresentable & CaseIterable>(_ key: String, _ type: T.Type) throws -> T
    where T.RawValue == String {
        let raw = try requiredString(key)
        guard let parsed = T(rawValue: raw) else {
            throw WGPUError.validation("unknown value \"\(raw)\" (allowed: \(Self.allowed(type)))", path: childPath(key))
        }
        return parsed
    }

    public func enumValue<T: RawRepresentable & CaseIterable>(_ key: String, default fallback: T) throws -> T
    where T.RawValue == String {
        guard has(key) else { return fallback }
        return try requiredEnum(key, T.self)
    }

    public func optionalEnum<T: RawRepresentable & CaseIterable>(_ key: String, _ type: T.Type) throws -> T?
    where T.RawValue == String {
        guard has(key) else { return nil }
        return try requiredEnum(key, type)
    }

    private static func allowed<T: RawRepresentable & CaseIterable>(_ type: T.Type) -> String
    where T.RawValue == String {
        T.allCases.map { "\($0.rawValue)" }.joined(separator: ", ")
    }

    public func flags<T: OptionSet>(_ key: String, _ type: T.Type, default fallback: T) throws -> T
    where T.RawValue == Int {
        guard let raw = try optionalInt(key) else { return fallback }
        return T(rawValue: raw)
    }

    public func requiredFlags<T: OptionSet>(_ key: String, _ type: T.Type) throws -> T where T.RawValue == Int {
        T(rawValue: try requiredInt(key))
    }

    // MARK: - Nested structures

    public func requiredObject(_ key: String) throws -> WGPUValueReader {
        guard let raw = value(key) else { throw missing(key) }
        guard let dictionary = raw as? [String: Any] else { throw mismatch(key, expected: "an object") }
        return WGPUValueReader(dictionary, path: childPath(key))
    }

    public func object(_ key: String) -> WGPUValueReader? {
        guard let dictionary = value(key) as? [String: Any] else { return nil }
        return WGPUValueReader(dictionary, path: childPath(key))
    }

    public func requiredObjects(_ key: String) throws -> [WGPUValueReader] {
        guard let raw = value(key) else { throw missing(key) }
        guard let array = raw as? [Any] else { throw mismatch(key, expected: "an array") }
        return try array.enumerated().map { index, element in
            guard let dictionary = element as? [String: Any] else {
                throw WGPUError.validation("an object is required", path: "\(childPath(key))[\(index)]")
            }
            return WGPUValueReader(dictionary, path: "\(childPath(key))[\(index)]")
        }
    }

    public func objects(_ key: String) throws -> [WGPUValueReader] {
        guard has(key) else { return [] }
        return try requiredObjects(key)
    }

    public func numbers(_ key: String) throws -> [Double] {
        guard let raw = value(key) else { throw missing(key) }
        guard let array = raw as? [Any] else { throw mismatch(key, expected: "a number array") }
        return try array.enumerated().map { index, element in
            guard let number = element as? NSNumber else {
                throw WGPUError.validation("a number is required", path: "\(childPath(key))[\(index)]")
            }
            return number.doubleValue
        }
    }

    public func integers(_ key: String) throws -> [Int] {
        guard let raw = value(key) else { throw missing(key) }
        guard let array = raw as? [Any] else { throw mismatch(key, expected: "an integer array") }
        return try array.enumerated().map { index, element in
            let path = "\(childPath(key))[\(index)]"
            guard let number = element as? NSNumber else {
                throw WGPUError.validation("an integer is required", path: path)
            }
            return try Self.enforceRange(number, path: path)
        }
    }

    /// A numeric map of the form `{ "name": value }` (pipeline `constants`).
    public func numberMap(_ key: String) -> [String: Double] {
        guard let raw = value(key) as? [String: Any] else { return [:] }
        return raw.compactMapValues { ($0 as? NSNumber)?.doubleValue }
    }

    public func handles(_ key: String) throws -> [WGPUHandle] {
        try integers(key).map(WGPUHandle.init)
    }

    /// String array. **`null` entries stay nil** — places such as a render bundle's `colorFormats`
    /// use null to mean "no attachment in this slot".
    public func strings(_ key: String) throws -> [String?] {
        guard let raw = value(key) else { return [] }
        guard let array = raw as? [Any] else { throw mismatch(key, expected: "a string array") }
        return try array.enumerated().map { index, element in
            if element is NSNull { return nil }
            guard let string = element as? String else {
                throw WGPUError.validation("a string is required", path: "\(childPath(key))[\(index)]")
            }
            return string
        }
    }

    // MARK: - Binary

    /// Reads a byte sequence. All three representations are accepted:
    ///
    /// - `NSData` — **the path the JS shim uses.** Lynx converted an `ArrayBuffer` into it.
    /// - a base64 string — for hand-written command streams (test harnesses, `RenderHarness.base64`).
    /// - a number array — for small debugging cases.
    public func requiredData(_ key: String) throws -> Data {
        guard let raw = value(key) else { throw missing(key) }
        if let data = raw as? Data { return data }
        if let string = raw as? String {
            guard let decoded = Data(base64Encoded: string) else {
                throw WGPUError.validation("base64 decoding failed", path: childPath(key))
            }
            return decoded
        }
        if let array = raw as? [Any] {
            var bytes = [UInt8]()
            bytes.reserveCapacity(array.count)
            for (index, element) in array.enumerated() {
                let path = "\(childPath(key))[\(index)]"
                guard let number = element as? NSNumber else {
                    throw WGPUError.validation("a byte (0...255) is required", path: path)
                }
                // Wraps like a JS `Uint8Array` store once the value is a real integer.
                bytes.append(UInt8(truncatingIfNeeded: try Self.enforceRange(number, path: path)))
            }
            return Data(bytes)
        }
        throw mismatch(key, expected: "an ArrayBuffer, base64 string or byte array")
    }

    /// Same as `requiredData` but nil when absent or unreadable.
    public func data(_ key: String) -> Data? {
        guard value(key) != nil else { return nil }
        return try? requiredData(key)
    }

    public func optionalData(_ key: String) throws -> Data? {
        guard has(key) else { return nil }
        return try requiredData(key)
    }

    // MARK: - Composite values

    /// Accepts both `{r,g,b,a}` and `[r,g,b,a]` (WebGPU `GPUColor`).
    public func color(_ key: String, default fallback: WGPUColor) throws -> WGPUColor {
        guard let raw = value(key) else { return fallback }
        if let dictionary = raw as? [String: Any] {
            let reader = WGPUValueReader(dictionary, path: childPath(key))
            return WGPUColor(
                red: reader.double("r", default: 0),
                green: reader.double("g", default: 0),
                blue: reader.double("b", default: 0),
                alpha: reader.double("a", default: 1)
            )
        }
        if let array = raw as? [Any] {
            let components = array.compactMap { ($0 as? NSNumber)?.doubleValue }
            guard components.count >= 3 else {
                throw WGPUError.validation("a color needs at least 3 components", path: childPath(key))
            }
            return WGPUColor(
                red: components[0],
                green: components[1],
                blue: components[2],
                alpha: components.count > 3 ? components[3] : 1
            )
        }
        throw mismatch(key, expected: "a color ({r,g,b,a} or [r,g,b,a])")
    }

    /// `{width,height,depthOrArrayLayers}` or `[w,h,d]` (WebGPU `GPUExtent3D`).
    public func requiredExtent(_ key: String) throws -> WGPUExtent3D {
        guard let raw = value(key) else { throw missing(key) }
        if let dictionary = raw as? [String: Any] {
            let reader = WGPUValueReader(dictionary, path: childPath(key))
            return WGPUExtent3D(
                width: try reader.requiredInt("width"),
                height: try reader.int("height", default: 1),
                depthOrArrayLayers: try reader.int("depthOrArrayLayers", default: 1)
            )
        }
        if let array = raw as? [Any] {
            let components = try integerComponents(array, key: key)
            guard let width = components.first else {
                throw WGPUError.validation("a size needs at least 1 component", path: childPath(key))
            }
            return WGPUExtent3D(
                width: width,
                height: components.count > 1 ? components[1] : 1,
                depthOrArrayLayers: components.count > 2 ? components[2] : 1
            )
        }
        throw mismatch(key, expected: "a size ({width,height,…} or [w,h,d])")
    }

    /// Same as `requiredExtent` but nil when absent — for callers that default to the source size.
    /// A present but unreadable size is an error, not "absent": falling back to the source size
    /// would silently copy more than was asked for.
    public func extent(_ key: String) throws -> WGPUExtent3D? {
        guard value(key) != nil else { return nil }
        return try requiredExtent(key)
    }

    /// `{x,y,z}` or `[x,y,z]` (WebGPU `GPUOrigin3D`). Absent means the origin.
    public func origin(_ key: String) throws -> WGPUOrigin3D {
        guard let raw = value(key) else { return WGPUOrigin3D() }
        if let dictionary = raw as? [String: Any] {
            let reader = WGPUValueReader(dictionary, path: childPath(key))
            return WGPUOrigin3D(
                x: try reader.int("x", default: 0),
                y: try reader.int("y", default: 0),
                z: try reader.int("z", default: 0)
            )
        }
        if let array = raw as? [Any] {
            let components = try integerComponents(array, key: key)
            return WGPUOrigin3D(
                x: components.count > 0 ? components[0] : 0,
                y: components.count > 1 ? components[1] : 0,
                z: components.count > 2 ? components[2] : 0
            )
        }
        throw mismatch(key, expected: "an origin ({x,y,z} or [x,y,z])")
    }

    /// The numbers of a `[w,h,d]` / `[x,y,z]` array form. Non-numbers are skipped as before;
    /// numbers go through `enforceRange` with their index in the path.
    private func integerComponents(_ array: [Any], key: String) throws -> [Int] {
        try array.enumerated().compactMap { index, element in
            guard let number = element as? NSNumber else { return nil }
            return try Self.enforceRange(number, path: "\(childPath(key))[\(index)]")
        }
    }
}

/// WebGPU `GPUColor`.
public struct WGPUColor: Equatable, Sendable {
    public var red: Double
    public var green: Double
    public var blue: Double
    public var alpha: Double

    public init(red: Double = 0, green: Double = 0, blue: Double = 0, alpha: Double = 1) {
        self.red = red
        self.green = green
        self.blue = blue
        self.alpha = alpha
    }

    public static let transparent = WGPUColor(red: 0, green: 0, blue: 0, alpha: 0)
    public static let black = WGPUColor(red: 0, green: 0, blue: 0, alpha: 1)
}

/// WebGPU `GPUExtent3D`.
public struct WGPUExtent3D: Equatable, Sendable {
    public var width: Int
    public var height: Int
    public var depthOrArrayLayers: Int

    public init(width: Int, height: Int = 1, depthOrArrayLayers: Int = 1) {
        self.width = width
        self.height = height
        self.depthOrArrayLayers = depthOrArrayLayers
    }
}

/// WebGPU `GPUOrigin3D`.
public struct WGPUOrigin3D: Equatable, Sendable {
    public var x: Int
    public var y: Int
    public var z: Int

    public init(x: Int = 0, y: Int = 0, z: Int = 0) {
        self.x = x
        self.y = y
        self.z = z
    }
}
