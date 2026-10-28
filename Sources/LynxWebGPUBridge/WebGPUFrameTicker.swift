#if canImport(Lynx)
import Foundation
import QuartzCore
import UIKit

/// `CADisplayLink`-based frame driver.
///
/// It always runs on the main thread and the callback only sends one global event (GPU work happens on the JS thread).
final class WebGPUFrameTicker {
    /// (timestamp in seconds, seconds since the last **delivered** frame) → whether this frame was delivered.
    ///
    /// A tick the host holds back (GPU in flight, JS not done with the previous frame) returns `false` and
    /// leaves the delta's origin where it was — so the next delivered frame's delta includes the skipped
    /// ticks, the time the receiver actually went without a frame.
    var onFrame: ((CFTimeInterval, CFTimeInterval) -> Bool)?

    private var displayLink: CADisplayLink?
    /// Timestamp of the last delivered tick, 0 before the first. Main thread only — `start`/`stop` reset it
    /// there, the same hop that swaps the link.
    private var lastDelivered: CFTimeInterval = 0

    func start(preferredFramesPerSecond: Int) {
        let begin = { [weak self] in
            guard let self else { return }
            self.displayLink?.invalidate()
            let link = CADisplayLink(target: WeakProxy(self), selector: #selector(WeakProxy.tick(_:)))
            if preferredFramesPerSecond > 0 {
                link.preferredFrameRateRange = CAFrameRateRange(
                    minimum: Float(max(preferredFramesPerSecond / 2, 1)),
                    maximum: Float(preferredFramesPerSecond),
                    preferred: Float(preferredFramesPerSecond)
                )
            }
            link.add(to: .main, forMode: .common)
            self.displayLink = link
            self.lastDelivered = 0
        }
        if Thread.isMainThread { begin() } else { DispatchQueue.main.async(execute: begin) }
    }

    func stop() {
        let end = { [weak self] in
            self?.displayLink?.invalidate()
            self?.displayLink = nil
            self?.lastDelivered = 0
        }
        if Thread.isMainThread { end() } else { DispatchQueue.main.async(execute: end) }
    }

    deinit {
        displayLink?.invalidate()
    }

    fileprivate func handle(_ link: CADisplayLink) {
        let delta = lastDelivered > 0 ? link.timestamp - lastDelivered : 0
        if onFrame?(link.timestamp, delta) == true { lastDelivered = link.timestamp }
    }

    /// CADisplayLink retains its target strongly, so a proxy breaks the reference cycle.
    private final class WeakProxy: NSObject {
        private weak var owner: WebGPUFrameTicker?
        init(_ owner: WebGPUFrameTicker) { self.owner = owner }

        @objc func tick(_ link: CADisplayLink) {
            owner?.handle(link)
        }
    }
}
#endif
