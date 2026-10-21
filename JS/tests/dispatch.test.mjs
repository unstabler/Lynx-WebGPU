/**
 * The `dispatchWorkgroups` payload — counts go on **as given**.
 *
 * `0` is a valid count that dispatches nothing (`Math.ceil(n / 64)` with `n = 0`); the shim used to send
 * `x || 1`, which ran the shader once. Only an omitted `y`/`z` defaults to `1`.
 */
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { installNativeMock, makeDevice, commandsOf } from './helpers.mjs';

/** The `dispatchWorkgroups` commands one compute pass sends for each argument list. */
async function dispatched(...calls) {
  const state = installNativeMock();
  const device = await makeDevice();
  const encoder = device.createCommandEncoder();
  const pass = encoder.beginComputePass();
  for (const args of calls) pass.dispatchWorkgroups(...args);
  pass.end();
  device.queue.submit([encoder.finish()]);
  return commandsOf(state)
    .filter((entry) => entry.op === 'dispatchWorkgroups')
    .map(({ x, y, z }) => [x, y, z]);
}

test('a zero count is sent as 0, not raised to 1', async () => {
  assert.deepEqual(
    await dispatched([0], [0, 0, 0], [4, 0], [4, 3, 0]),
    [[0, 1, 1], [0, 0, 0], [4, 0, 1], [4, 3, 0]]
  );
});

test('an omitted y/z defaults to 1', async () => {
  assert.deepEqual(await dispatched([5], [5, 2], [5, 2, 3]), [[5, 1, 1], [5, 2, 1], [5, 2, 3]]);
});
