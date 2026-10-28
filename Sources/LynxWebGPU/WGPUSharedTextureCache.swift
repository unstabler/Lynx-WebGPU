import Foundation
import Metal
import LynxWebGPUCore

/// One decoded and uploaded texture, shared by every context on the same device that imported its key.
///
/// **Ownership.** Each context's registry entry holds this object strongly (through `WGPUTextureObject`
/// and the views made from it); the cache holds it only weakly. When the last context lets go —
/// `destroy`, `device.destroy()`, GC — `deinit` removes the cache entry, and the next request for the
/// key decodes again. One context resetting therefore never takes the texture from another.
public final class WGPUSharedTexture {
    public let texture: MTLTexture
    /// The request that made it. Later requests under the same key get this texture whatever they asked
    /// for (the key contract — `WGPUSharedTextureRequest`).
    public let request: WGPUSharedTextureRequest
    public var width: Int { texture.width }
    public var height: Int { texture.height }

    fileprivate let slot: WGPUSharedTextureCache.SlotKey
    fileprivate weak var cache: WGPUSharedTextureCache?

    fileprivate init(texture: MTLTexture, request: WGPUSharedTextureRequest,
                     slot: WGPUSharedTextureCache.SlotKey, cache: WGPUSharedTextureCache) {
        self.texture = texture
        self.request = request
        self.slot = slot
        self.cache = cache
    }

    deinit {
        cache?.release(slot)
    }
}

/// The process-wide cache behind `importSharedTexture` — one texture per (device, key).
///
/// **Why it lives in `LynxWebGPU`.** A host that links this target dynamically gets one copy of
/// `shared` for every module in the process. The bridge sources are compiled into app targets (and
/// vendored by some), so a cache there would split into one per copy.
///
/// **Keyed by device.** Contexts on different `MTLDevice`s never share an entry — a texture belongs to
/// its device. Inject one device into every context (`LynxWebGPUContext(device:)`) to share across them;
/// a consumer running its own device stays separate and safe.
///
/// **What it holds.** Only textures some context still uses (weakly), plus requests in flight.
/// Concurrent requests for a key wait on the one decode; a failure reaches every waiter with the same
/// error and leaves no entry, so the next request tries again. There is nothing to drop on a memory
/// warning — the owners are the contexts.
public final class WGPUSharedTextureCache {
    public static let shared = WGPUSharedTextureCache()

    struct SlotKey: Hashable {
        let device: ObjectIdentifier
        let key: String
    }

    private final class WeakEntry {
        weak var value: WGPUSharedTexture?
        init(_ value: WGPUSharedTexture) { self.value = value }
    }

    private enum Slot {
        /// The first request (which decides the pixels) and everyone waiting for it.
        case loading(WGPUSharedTextureRequest, [(Result<WGPUSharedTexture, WGPUError>) -> Void])
        case ready(WeakEntry)
    }

    private var slots: [SlotKey: Slot] = [:]
    private var makeRuns = 0
    private let lock = NSLock()

    /// Called when a key comes back asking for different pixels than the request that made its entry
    /// (`describesSamePixels`). The first texture is returned either way — this only makes the broken
    /// key contract visible. Logs in debug builds; tests swap it.
    var warnConflict: (String) -> Void = { message in
        #if DEBUG
        WGPULog.runtime.warning("\(message, privacy: .public)")
        #endif
    }

    init() {}

    /// Entries some context still holds. Requests in flight are not counted.
    public var liveEntryCount: Int {
        lock.lock()
        defer { lock.unlock() }
        return slots.values.reduce(0) { count, slot in
            if case .ready(let weak) = slot, weak.value != nil { return count + 1 }
            return count
        }
    }

    /// How many times a texture was actually made — the test probe for "decoded once".
    var makeCount: Int {
        lock.lock()
        defer { lock.unlock() }
        return makeRuns
    }

    /// The live entry for a key, if any — a test probe.
    func entry(forKey key: String, device: MTLDevice) -> WGPUSharedTexture? {
        lock.lock()
        defer { lock.unlock() }
        guard case .ready(let weak)? = slots[SlotKey(device: ObjectIdentifier(device), key: key)] else { return nil }
        return weak.value
    }

    /// Hands back the texture for `request.key` on `device`, making it with `make` only when no context
    /// holds one and no request is already making it.
    ///
    /// - Parameter make: decodes and uploads; runs on a background queue, at most once per miss, and
    ///   must return a texture whose upload has **completed** (other contexts read it from their own
    ///   queues straight away).
    /// - Parameter completion: exactly once, from any thread. The caller must keep the returned object
    ///   for as long as it uses the texture — the cache does not.
    func acquire(
        _ request: WGPUSharedTextureRequest,
        device: MTLDevice,
        make: @escaping () throws -> MTLTexture,
        completion: @escaping (Result<WGPUSharedTexture, WGPUError>) -> Void
    ) {
        let slotKey = SlotKey(device: ObjectIdentifier(device), key: request.key)
        lock.lock()
        if case .ready(let weak)? = slots[slotKey], let entry = weak.value {
            lock.unlock()
            warnIfConflicting(entry.request, request)
            completion(.success(entry))
            return
        }
        if case .loading(let first, let waiters)? = slots[slotKey] {
            slots[slotKey] = .loading(first, waiters + [completion])
            lock.unlock()
            warnIfConflicting(first, request)
            return
        }
        // A miss — or an entry whose last owner is releasing it right now; its `deinit` finds the new
        // slot and leaves it alone.
        slots[slotKey] = .loading(request, [completion])
        makeRuns += 1
        lock.unlock()

        DispatchQueue.global(qos: .userInitiated).async {
            let made: Result<MTLTexture, WGPUError>
            do {
                made = .success(try make())
            } catch let error as WGPUError {
                made = .failure(error)
            } catch {
                made = .failure(WGPUError.backend("\(error)"))
            }
            self.finish(slotKey, request: request, made: made)
        }
    }

    private func finish(_ slotKey: SlotKey, request: WGPUSharedTextureRequest, made: Result<MTLTexture, WGPUError>) {
        lock.lock()
        guard case .loading(_, let waiters)? = slots[slotKey] else {
            lock.unlock()
            return
        }
        let outcome: Result<WGPUSharedTexture, WGPUError>
        switch made {
        case .success(let texture):
            let entry = WGPUSharedTexture(texture: texture, request: request, slot: slotKey, cache: self)
            slots[slotKey] = .ready(WeakEntry(entry))
            outcome = .success(entry)
        case .failure(let error):
            // No entry — the next request decodes again (a missing file may have arrived since).
            slots[slotKey] = nil
            outcome = .failure(error)
        }
        lock.unlock()
        // Outside the lock: a waiter that does not keep the entry releases it, and `deinit` takes the lock.
        for waiter in waiters { waiter(outcome) }
    }

    fileprivate func release(_ slotKey: SlotKey) {
        lock.lock()
        if case .ready(let weak)? = slots[slotKey], weak.value == nil {
            slots[slotKey] = nil
        }
        lock.unlock()
    }

    private func warnIfConflicting(_ existing: WGPUSharedTextureRequest, _ incoming: WGPUSharedTextureRequest) {
        guard !existing.describesSamePixels(as: incoming) else { return }
        func describe(_ request: WGPUSharedTextureRequest) -> String {
            let width = request.resizeWidth.map(String.init) ?? "auto"
            let height = request.resizeHeight.map(String.init) ?? "auto"
            return "'\(request.name)' at \(width)x\(height)"
        }
        warnConflict(
            "importSharedTexture: key '\(incoming.key)' was made from \(describe(existing)) and is now asked "
                + "for \(describe(incoming)) — the first texture is returned. Put everything that changes "
                + "the pixels (source, version, size) into the key."
        )
    }
}
