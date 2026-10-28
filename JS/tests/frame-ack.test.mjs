/**
 * The frame tick fan-out and its ack — rAF-style backpressure toward the native ticker.
 *
 * Native sends the next `webgpu:frame` only after the shim acknowledges the previous one
 * (`frameHandled`). Otherwise, whenever the JS thread falls behind the refresh rate, frame events
 * pile up in the JS message queue without bound and touch events queue behind them — input lag
 * grows to seconds. So the shim must ack **exactly once at the end of every tick**: after the
 * presents, on idle ticks too, and even when the frame callback throws.
 *
 * Every `startFrameLoop` on a view shares that one ticker. The shim fans each tick out to the
 * subscribers that are due — the link runs at the highest fps asked for, each subscriber keeps its
 * own rate and measures its own `delta` — and the tick still acks once, however many ran in it.
 */
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { installNativeMock, makeDevice, installFrameEmitter } from './helpers.mjs';
import { startFrameLoop } from '../webgpu.js';

/** Lays the frame emitter, runs the body, and always removes it. */
function withFrames(body) {
  return async () => {
    const emitter = installFrameEmitter();
    try {
      await body(emitter);
    } finally {
      emitter.uninstall();
    }
  };
}

/** One piece of GPU work inside a frame. */
function doSomeWork(device) {
  device.queue.writeBuffer(device.createBuffer({ size: 4, usage: 0x0008 }), 0, new Uint8Array(4));
  device.queue.submit([]);
}

test('every tick acks exactly once — an idle tick too', withFrames(async (emitter) => {
  const state = installNativeMock();
  const stop = startFrameLoop(() => {});
  try {
    emitter.tick();
    assert.equal(state.frameHandledCalls, 1, 'an idle tick must still release the native gate');
    emitter.tick();
    assert.equal(state.frameHandledCalls, 2);
  } finally {
    stop();
  }
}));

test('the ack leaves after the present', withFrames(async (emitter) => {
  const state = installNativeMock();
  const device = await makeDevice();

  // Record the bridge call order — the ack marks "the frame is fully handled", so it must not
  // overtake the present flush.
  /** @type {string[]} */
  const sequence = [];
  const module = globalThis.NativeModules.WebGPU;
  const execute = module.execute.bind(module);
  module.execute = (payload) => {
    sequence.push(payload.present === false ? 'execute' : 'present');
    return execute(payload);
  };
  const frameHandled = module.frameHandled.bind(module);
  module.frameHandled = () => {
    sequence.push('ack');
    return frameHandled();
  };

  const stop = startFrameLoop(() => doSomeWork(device));
  try {
    sequence.length = 0;
    emitter.tick();
  } finally {
    stop();
  }

  assert.equal(state.frameHandledCalls, 1);
  assert.equal(sequence[sequence.length - 1], 'ack', `the ack must come last (order: ${sequence})`);
  assert.ok(sequence.includes('present'), `the tick must have presented (order: ${sequence})`);
}));

test('a throwing frame callback still acks', withFrames(async (emitter) => {
  const state = installNativeMock();
  const stop = startFrameLoop(() => {
    throw new Error('scene bug');
  });
  try {
    assert.throws(() => emitter.tick(), /scene bug/);
    assert.equal(state.frameHandledCalls, 1, 'the gate must be released even when the callback throws');
  } finally {
    stop();
  }
}));

test('an old native without frameHandled does not break the tick', withFrames(async (emitter) => {
  const state = installNativeMock();
  const device = await makeDevice();
  delete globalThis.NativeModules.WebGPU.frameHandled;

  const stop = startFrameLoop(() => doSomeWork(device));
  try {
    emitter.tick();   // must not throw
  } finally {
    stop();
  }
  const presents = state.executeCalls.filter((call) => call.present !== false);
  assert.equal(presents.length, 1, 'the frame itself must still go through');
}));

test('after the loop stops, ticks no longer ack', withFrames(async (emitter) => {
  const state = installNativeMock();
  const stop = startFrameLoop(() => {});
  emitter.tick();
  assert.equal(state.frameHandledCalls, 1);
  stop();
  // A tick that was already queued when the loop stopped finds no listener — and the shim must
  // not ack from anywhere else (the native side clears its own gate on stopFrameLoop).
  emitter.tick();
  assert.equal(state.frameHandledCalls, 1);
  assert.equal(state.stopFrameLoopCalls, 1);
}));

// --- The fan-out ------------------------------------------------------------------------------

/** One 60 Hz display period in ms. */
const PERIOD = 1000 / 60;

/** `actual` within a float's rounding of `expected` — timestamps here are fractional ms. */
function assertClose(actual, expected, message) {
  assert.ok(Math.abs(actual - expected) < 1e-6, `${message} (expected ${expected}, got ${actual})`);
}

test('the link runs at the fastest subscriber and a 30 fps one takes every other 60 fps tick', withFrames(async (emitter) => {
  const state = installNativeMock();
  /** @type {{timestamp: number, delta: number}[]} */
  const slow = [];
  /** @type {{timestamp: number, delta: number}[]} */
  const fast = [];
  const stopSlow = startFrameLoop((frame) => slow.push(frame), { fps: 30 });
  const stopFast = startFrameLoop((frame) => fast.push(frame), { fps: 60 });
  try {
    assert.deepEqual(state.startFrameLoopCalls.map((call) => call.fps), [30, 60],
      'the link must be retuned up to the fastest subscriber');

    // Display link timestamps jitter around the period — a tick landing a hair early must still count,
    // or the 30 fps subscriber would drop to 20.
    const ticks = [1000, 1016.9, 1033.1, 1050.2, 1066.5, 1083.4, 1099.8];
    for (const timestamp of ticks) emitter.tick({ timestamp, delta: PERIOD });

    assert.deepEqual(fast.map((frame) => frame.timestamp), ticks, 'the fastest subscriber takes every tick');
    assert.deepEqual(slow.map((frame) => frame.timestamp), [1000, 1033.1, 1066.5, 1099.8],
      'the 30 fps subscriber takes every other tick');
    // Each delta counts from **that subscriber's** previous call — 0 on the first.
    assert.equal(slow[0].delta, 0);
    assertClose(slow[1].delta, 33.1, 'the slow delta spans the tick it skipped');
    assertClose(fast[1].delta, 16.9, 'the fast delta is one tick');
    assert.equal(state.frameHandledCalls, ticks.length, 'one ack per tick, filtered or not');
  } finally {
    stopFast();
    stopSlow();
  }
}));

test('when the fast subscriber leaves, the link drops back to the one that remains', withFrames(async () => {
  const state = installNativeMock();
  const stopSlow = startFrameLoop(() => {}, { fps: 30 });
  const stopFast = startFrameLoop(() => {}, { fps: 60 });
  // Another 30 fps subscriber changes nothing — a restart re-creates the display link, so it is not done for nothing.
  const stopOther = startFrameLoop(() => {}, { fps: 30 });
  assert.deepEqual(state.startFrameLoopCalls.map((call) => call.fps), [30, 60]);

  stopFast();
  assert.deepEqual(state.startFrameLoopCalls.map((call) => call.fps), [30, 60, 30],
    'the link must not stay at the burst rate after the burst ends');
  assert.equal(state.stopFrameLoopCalls, 0, 'the link keeps running for the subscribers that remain');

  stopOther();
  assert.equal(state.startFrameLoopCalls.length, 3, 'the highest fps did not change');
  stopSlow();
  assert.equal(state.stopFrameLoopCalls, 1, 'the last subscriber stops the link');
}));

test('one tick with three subscribers acks once and presents once', withFrames(async (emitter) => {
  const state = installNativeMock();
  const device = await makeDevice();
  /** @type {string[]} */
  const ran = [];
  const stops = [
    startFrameLoop(() => { ran.push('a'); doSomeWork(device); }),
    startFrameLoop(() => { ran.push('b'); doSomeWork(device); }),
    startFrameLoop(() => { ran.push('c'); doSomeWork(device); }, { fps: 30 }),
  ];
  try {
    state.executeCalls.length = 0;
    emitter.tick({ timestamp: 1000, delta: PERIOD });
  } finally {
    for (const stop of stops) stop();
  }

  assert.deepEqual(ran, ['a', 'b', 'c']);
  assert.equal(state.frameHandledCalls, 1, 'the native gate must see one tick, not three');
  const presents = state.executeCalls.filter((call) => call.present !== false);
  assert.equal(presents.length, 1, `one present per tick (${presents.length} went out)`);
}));

test('a delta covers the ticks native skipped under backpressure', withFrames(async (emitter) => {
  installNativeMock();
  /** @type {number[]} */
  const deltas = [];
  const stop = startFrameLoop(({ delta }) => deltas.push(delta));
  try {
    emitter.tick({ timestamp: 1000, delta: 0 });
    emitter.tick({ timestamp: 1000 + PERIOD, delta: PERIOD });
    // Native held back three ticks waiting for the ack. The delta it sends was measured from its own
    // previous link tick — the shim must not trust it.
    emitter.tick({ timestamp: 1000 + 5 * PERIOD, delta: PERIOD });
  } finally {
    stop();
  }
  assert.equal(deltas[0], 0, 'the first call has nothing to measure from');
  assertClose(deltas[1], PERIOD, 'one tick');
  assertClose(deltas[2], 4 * PERIOD, 'the skipped ticks belong in the delta');
}));

test('a throwing subscriber does not starve the others, and the tick still acks once', withFrames(async (emitter) => {
  const state = installNativeMock();
  /** @type {string[]} */
  const ran = [];
  const stopThrowing = startFrameLoop(() => {
    ran.push('throwing');
    throw new Error('scene bug');
  });
  const stopOther = startFrameLoop(() => ran.push('other'));
  try {
    assert.throws(() => emitter.tick(), /scene bug/, 'the error is not swallowed');
    assert.deepEqual(ran, ['throwing', 'other']);
    assert.equal(state.frameHandledCalls, 1);
  } finally {
    stopThrowing();
    stopOther();
  }
}));

test('stopping inside a tick — oneself or another — and subscribing inside a tick', withFrames(async (emitter) => {
  const state = installNativeMock();
  /** @type {string[]} */
  const ran = [];
  /** @type {() => void} */
  let stopB = () => {};
  /** @type {() => void} */
  let stopLate = () => {};
  let subscribed = false;
  const stopA = startFrameLoop(() => {
    ran.push('a');
    stopB();   // b comes later in this tick — it must not run
    stopA();
    if (!subscribed) {
      subscribed = true;
      stopLate = startFrameLoop(() => ran.push('late'));   // belongs to the next tick, as with rAF
    }
  });
  stopB = startFrameLoop(() => ran.push('b'));
  const stopC = startFrameLoop(() => ran.push('c'));
  try {
    emitter.tick({ timestamp: 1000, delta: PERIOD });
    assert.deepEqual(ran, ['a', 'c']);
    emitter.tick({ timestamp: 1000 + PERIOD, delta: PERIOD });
    assert.deepEqual(ran, ['a', 'c', 'c', 'late']);
    assert.equal(state.frameHandledCalls, 2);
    assert.equal(state.stopFrameLoopCalls, 0, 'c and late still hold the link');
  } finally {
    stopC();
    stopLate();
  }
  assert.equal(state.stopFrameLoopCalls, 1);
}));
