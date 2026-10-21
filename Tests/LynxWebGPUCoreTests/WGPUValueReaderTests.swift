import XCTest
@testable import LynxWebGPUCore

/// Command stream decoding — verified by reproducing exactly what JS sends.
final class WGPUValueReaderTests: XCTestCase {
    func test_scalarReadsAndDefaults() throws {
        let reader = WGPUValueReader(["size": 128, "label": "vertices", "flag": true])

        XCTAssertEqual(try reader.requiredInt("size"), 128)
        XCTAssertEqual(try reader.int("missing", default: 7), 7)
        XCTAssertEqual(try reader.requiredString("label"), "vertices")
        XCTAssertTrue(reader.bool("flag", default: false))
        XCTAssertFalse(reader.bool("missing", default: false))
    }

    func test_aMissingRequiredFieldIsAnErrorWithAPath() {
        let reader = WGPUValueReader(["a": 1], path: "commands[3]")

        XCTAssertThrowsError(try reader.requiredInt("size")) { error in
            let wgpu = error as? WGPUError
            XCTAssertEqual(wgpu?.kind, .validation)
            XCTAssertEqual(wgpu?.path, "commands[3].size")
        }
    }

    func test_nullIsTreatedAsAbsent() {
        // JS's `undefined` arrives as NSNull after crossing Lynx.
        let reader = WGPUValueReader(["label": NSNull()])
        XCTAssertFalse(reader.has("label"))
        XCTAssertNil(reader.optionalString("label"))
    }

    func test_enumsParseBySpecSpellingAndListCandidatesWhenUnknown() throws {
        let reader = WGPUValueReader(["format": "bgra8unorm", "topology": "triangle-strip"])

        XCTAssertEqual(try reader.requiredEnum("format", WGPUTextureFormat.self), .bgra8unorm)
        XCTAssertEqual(try reader.requiredEnum("topology", WGPUPrimitiveTopology.self), .triangleStrip)

        let bad = WGPUValueReader(["format": "rgba8"])
        XCTAssertThrowsError(try bad.requiredEnum("format", WGPUTextureFormat.self)) { error in
            let message = (error as? WGPUError)?.message ?? ""
            XCTAssertTrue(message.contains("rgba8unorm"), "no list of possible values: \(message)")
        }
    }

    func test_readsBitmaskFlags() throws {
        let reader = WGPUValueReader(["usage": 0x0020 | 0x0008])
        let usage = try reader.requiredFlags("usage", WGPUBufferUsage.self)

        XCTAssertTrue(usage.contains(.vertex))
        XCTAssertTrue(usage.contains(.copyDst))
        XCTAssertFalse(usage.contains(.uniform))
    }

    func test_colorAcceptsBothObjectAndArrayForms() throws {
        let object = WGPUValueReader(["clearValue": ["r": 1.0, "g": 0.5, "b": 0.0, "a": 1.0]])
        XCTAssertEqual(try object.color("clearValue", default: .black), WGPUColor(red: 1, green: 0.5, blue: 0, alpha: 1))

        let array = WGPUValueReader(["clearValue": [1.0, 0.5, 0.0]])
        XCTAssertEqual(try array.color("clearValue", default: .black), WGPUColor(red: 1, green: 0.5, blue: 0, alpha: 1))

        let missing = WGPUValueReader([:])
        XCTAssertEqual(try missing.color("clearValue", default: .black), .black)
    }

    func test_extentAcceptsBothObjectAndArrayForms() throws {
        let object = WGPUValueReader(["size": ["width": 64, "height": 32]])
        XCTAssertEqual(try object.requiredExtent("size"), WGPUExtent3D(width: 64, height: 32))

        let array = WGPUValueReader(["size": [64, 32, 6]])
        XCTAssertEqual(try array.requiredExtent("size"), WGPUExtent3D(width: 64, height: 32, depthOrArrayLayers: 6))
    }

    func test_binaryAcceptsDataBase64AndByteArrays() throws {
        let bytes: [UInt8] = [0xDE, 0xAD, 0xBE, 0xEF]

        // The path the JS shim uses — what Lynx turned from an ArrayBuffer into NSData arrives as-is.
        let native = WGPUValueReader(["data": Data(bytes)])
        XCTAssertEqual(Array(try native.requiredData("data")), bytes)

        // The same when it arrives as NSData (the type the bridge actually passes).
        let bridged = WGPUValueReader(["data": NSData(data: Data(bytes))])
        XCTAssertEqual(Array(try bridged.requiredData("data")), bytes)

        let base64 = WGPUValueReader(["data": Data(bytes).base64EncodedString()])
        XCTAssertEqual(Array(try base64.requiredData("data")), bytes)

        let array = WGPUValueReader(["data": bytes.map { Int($0) }])
        XCTAssertEqual(Array(try array.requiredData("data")), bytes)

        let broken = WGPUValueReader(["data": "!!not base64!!"])
        XCTAssertThrowsError(try broken.requiredData("data"))
    }

    func test_aNestedArrayPathKeepsTheIndex() {
        let reader = WGPUValueReader(["entries": [["binding": 0], ["nope": 1]]], path: "cmd")
        let entries = try? reader.requiredObjects("entries")

        XCTAssertEqual(entries?.count, 2)
        XCTAssertThrowsError(try entries?[1].requiredInt("binding")) { error in
            XCTAssertEqual((error as? WGPUError)?.path, "cmd.entries[1].binding")
        }
    }

    // MARK: - Integer range (WebIDL [EnforceRange])

    /// What a JS number can be that `Int(Double)` traps on (or `NSNumber.intValue` saturates/wraps).
    private static let unusableIntegers: [Double] = [.nan, .infinity, -.infinity, 1e20, -1e20]

    private func assertValidation(
        _ expression: @autoclosure () throws -> Any?, path: String,
        file: StaticString = #filePath, line: UInt = #line
    ) {
        XCTAssertThrowsError(try expression(), file: file, line: line) { error in
            guard let error = error as? WGPUError else {
                return XCTFail("not a WGPUError: \(error)", file: file, line: line)
            }
            XCTAssertEqual(error.kind, .validation, file: file, line: line)
            XCTAssertEqual(error.path, path, file: file, line: line)
        }
    }

    func test_integersRejectNonFiniteAndUnsafeValuesWithTheElementPath() {
        for bad in Self.unusableIntegers {
            let reader = WGPUValueReader(["offsets": [256, bad]], path: "commands[2]")
            assertValidation(try reader.integers("offsets"), path: "commands[2].offsets[1]")
            assertValidation(try reader.handles("offsets"), path: "commands[2].offsets[1]")
        }
    }

    func test_everyIntegerAccessorRejectsNonFiniteAndUnsafeValues() {
        for bad in Self.unusableIntegers {
            let scalar = WGPUValueReader(["v": bad], path: "cmd")
            assertValidation(try scalar.requiredInt("v"), path: "cmd.v")
            assertValidation(try scalar.int("v", default: 0), path: "cmd.v")
            assertValidation(try scalar.optionalInt("v"), path: "cmd.v")
            assertValidation(try scalar.requiredHandle("v"), path: "cmd.v")
            assertValidation(try scalar.optionalHandle("v"), path: "cmd.v")
            assertValidation(try scalar.flags("v", WGPUBufferUsage.self, default: []), path: "cmd.v")
            assertValidation(try scalar.requiredFlags("v", WGPUBufferUsage.self), path: "cmd.v")

            let objectForms = WGPUValueReader(["size": ["width": 4, "height": bad], "origin": ["z": bad]], path: "cmd")
            assertValidation(try objectForms.requiredExtent("size"), path: "cmd.size.height")
            assertValidation(try objectForms.extent("size"), path: "cmd.size.height")
            assertValidation(try objectForms.origin("origin"), path: "cmd.origin.z")

            let arrayForms = WGPUValueReader(["size": [4, 4, bad], "origin": [bad], "data": [1, bad]], path: "cmd")
            assertValidation(try arrayForms.requiredExtent("size"), path: "cmd.size[2]")
            assertValidation(try arrayForms.extent("size"), path: "cmd.size[2]")
            assertValidation(try arrayForms.origin("origin"), path: "cmd.origin[0]")
            assertValidation(try arrayForms.requiredData("data"), path: "cmd.data[1]")
        }
    }

    func test_integerReadsTruncateWithinTheSafeIntegerRange() throws {
        let max = Double(WGPUValueReader.maxSafeInteger)
        let reader = WGPUValueReader([
            "max": max, "min": -max, "over": max + 1, "under": -max - 1,
            "fraction": 1.9, "negativeFraction": -1.9, "flag": true,
            "int64": NSNumber(value: Int64.max), "text": "12",
        ])

        XCTAssertEqual(try reader.requiredInt("max"), WGPUValueReader.maxSafeInteger)
        XCTAssertEqual(try reader.requiredInt("min"), -WGPUValueReader.maxSafeInteger)
        assertValidation(try reader.requiredInt("over"), path: "over")
        assertValidation(try reader.requiredInt("under"), path: "under")
        assertValidation(try reader.requiredInt("int64"), path: "int64")
        // [EnforceRange] truncates toward zero — the same result `intValue` gave before.
        XCTAssertEqual(try reader.requiredInt("fraction"), 1)
        XCTAssertEqual(try reader.requiredInt("negativeFraction"), -1)
        XCTAssertEqual(try reader.requiredInt("flag"), 1)
        // A non-number is still "absent" for the defaulted accessors (unchanged tolerance).
        XCTAssertEqual(try reader.int("text", default: 7), 7)
        XCTAssertNil(try reader.optionalInt("text"))
        XCTAssertNil(try reader.extent("missing"))
    }
}
