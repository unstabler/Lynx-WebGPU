import Foundation

/// Runs the transpiler's recursive passes on a thread whose stack is sized for them.
///
/// Parsing, reflection and MSL emission all recurse over the shader's nesting. The parser caps the
/// nesting (`WGSLParser.maximumNestingDepth`), but how much stack that takes is up to the emitter's
/// frames — several KB per nested call or block in a debug build — while the caller's stack is
/// whatever thread it happens to run on: commands run on the Lynx JS thread, and an iOS secondary
/// thread gets 512 KB by default. A shader the parser accepts must not then overflow the emitter.
///
/// So the entry points that take untrusted source — `WGSLShaderModule.init` and `translateToMSL` —
/// hop onto a dedicated thread with a 16 MB stack and wait for it. With the depth capped, that is far
/// more than any accepted shader needs, whatever the caller's stack. Module and pipeline creation are
/// not hot paths, so a thread per call costs nothing that matters. (The alternative, making every
/// walk iterative, would touch every pass and still leave the next one to get wrong.)
enum WGSLDeepStack {
    static let stackSize = 16 << 20

    /// Runs `work` to completion on the deep-stack thread, returning its value or rethrowing its error.
    ///
    /// `work` is escaping because the `Thread` keeps its block until the thread object goes away,
    /// which can be after this returns — the result itself is handed over before `finished` signals.
    static func run<T>(_ work: @escaping () throws -> T) throws -> T {
        let box = ResultBox<T>()
        let finished = DispatchSemaphore(value: 0)
        let thread = Thread {
            box.result = Result { try work() }
            finished.signal()
        }
        thread.stackSize = stackSize
        thread.start()
        finished.wait()
        guard let result = box.result else { throw DeepStackError.noResult }
        return try result.get()
    }

    private final class ResultBox<T> {
        var result: Result<T, Error>?
    }

    private enum DeepStackError: Error {
        case noResult
    }
}
