/**
 * `importSharedTexture()` — a process-wide shared texture, one per key, read-only in every page.
 *
 * The decoding, the sharing and the write refusal are native (Swift `SharedTextureTests`). What the shim
 * owns is checked here: what crosses the bridge, the `GPUTexture` it hands back (its usage tells web code
 * it cannot write), and that releasing it goes down the same `destroy` path as any texture.
 */
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { installNativeMock, makeDevice, commandsOf } from './helpers.mjs';
import { importSharedTexture, GPUTexture, GPUTextureUsage, GPUValidationError } from '../webgpu.js';

test('sends the key, url, size and label under a fresh handle', async () => {
  const state = installNativeMock({
    importSharedTextureResult: { ok: true, width: 1024, height: 1190, format: 'rgba8unorm' },
  });
  const device = await makeDevice();
  const other = device.createBuffer({ size: 16, usage: 0x0040 });

  const texture = await importSharedTexture(device, {
    key: 'clouds-v3@1024w', url: 'file:///themes/clouds.png', resizeWidth: 1024, label: 'clouds',
  });

  assert.equal(state.importSharedTextureCalls.length, 1);
  const params = state.importSharedTextureCalls[0];
  assert.deepEqual(params, {
    id: texture.id, key: 'clouds-v3@1024w', url: 'file:///themes/clouds.png', resizeWidth: 1024, label: 'clouds',
  }, 'omitted fields are not sent — native derives the other axis');
  assert.notEqual(texture.id, other.id, 'the handle comes from the module-wide counter');
});

test('hands back a GPUTexture with the shape native reports', async () => {
  installNativeMock({ importSharedTextureResult: { ok: true, width: 640, height: 480, format: 'rgba8unorm' } });
  const device = await makeDevice();

  const texture = await importSharedTexture(device, { key: 'k', url: 'a.png', label: 'bg' });

  assert.ok(texture instanceof GPUTexture);
  assert.equal(texture.width, 640);
  assert.equal(texture.height, 480);
  assert.equal(texture.depthOrArrayLayers, 1);
  assert.equal(texture.format, 'rgba8unorm');
  assert.equal(texture.dimension, '2d');
  assert.equal(texture.label, 'bg');
});

test('its usage is read-only — no COPY_DST, RENDER_ATTACHMENT or STORAGE_BINDING', async () => {
  installNativeMock();
  const device = await makeDevice();
  const texture = await importSharedTexture(device, { key: 'k', url: 'a.png' });

  assert.equal(texture.usage, GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_SRC);
  for (const write of ['COPY_DST', 'RENDER_ATTACHMENT', 'STORAGE_BINDING']) {
    assert.equal(texture.usage & GPUTextureUsage[write], 0, `${write} must not be set`);
  }
});

test('a write the native side refuses reaches the app as a GPUValidationError', async () => {
  // Native rejects every write to a shared texture (Swift `SharedTextureTests`) — the shim does not
  // second-guess it, so the error arrives in order through the usual channel.
  installNativeMock({
    executeResult: (payload) => ({
      ok: false,
      commandCount: payload.commands.length,
      errors: [{ kind: 'validation', message: 'writeTexture cannot write to a shared texture — it is read-only (importSharedTexture)' }],
    }),
  });
  const device = await makeDevice();
  const texture = await importSharedTexture(device, { key: 'k', url: 'a.png' });
  /** @type {any[]} */
  const seen = [];
  device.addEventListener('uncapturederror', (event) => seen.push(event.error));

  device.queue.writeTexture({ texture }, new Uint8Array(16), { bytesPerRow: 8 }, [2, 2]);
  device.queue.submit([]);

  assert.equal(seen.length, 1);
  assert.ok(seen[0] instanceof GPUValidationError);
  assert.match(seen[0].message, /read-only/);
});

test('destroy() sends this page\'s release once — the same path as any texture', async () => {
  const state = installNativeMock();
  const device = await makeDevice();
  const texture = await importSharedTexture(device, { key: 'k', url: 'a.png' });

  texture.destroy();
  device.queue.submit([]);

  const destroys = commandsOf(state).filter((command) => command.op === 'destroy' && command.id === texture.id);
  assert.equal(destroys.length, 1);
});

test('a lost wrapper is released by GC like any texture', {
  skip: !(typeof globalThis.gc === 'function' && typeof FinalizationRegistry === 'function'),
}, async () => {
  const state = installNativeMock();
  const device = await makeDevice();
  /** @type {number[]} */
  const ids = [];
  for (let index = 0; index < 20; index += 1) {
    ids.push((await importSharedTexture(device, { key: `k${index}`, url: 'a.png' })).id);
  }

  let released = false;
  for (let round = 0; round < 50 && !released; round += 1) {
    globalThis.gc();
    await new Promise((resolve) => setTimeout(resolve, 5));
    device.queue.submit([]);
    released = state.executeCalls.some((call) =>
      call.commands.some((command) => command.op === 'destroy' && ids.includes(command.id)));
  }
  assert.ok(released, 'an unreachable shared texture must release this page\'s reference');
});

test('a native failure rejects with its message', async () => {
  installNativeMock({
    importSharedTextureResult: {
      ok: false, errors: [{ kind: 'validation', message: "importSharedTexture could not decode 'a.png'" }],
    },
  });
  const device = await makeDevice();

  await assert.rejects(() => importSharedTexture(device, { key: 'k', url: 'a.png' }), /could not decode/);
});

test('a missing key or url is a TypeError before anything crosses the bridge', async () => {
  const state = installNativeMock();
  const device = await makeDevice();

  for (const descriptor of [{ url: 'a.png' }, { key: '', url: 'a.png' }, { key: 'k' }, { key: 'k', url: '' }]) {
    await assert.rejects(
      () => importSharedTexture(device, /** @type {any} */ (descriptor)), TypeError, JSON.stringify(descriptor)
    );
  }
  assert.equal(state.importSharedTextureCalls.length, 0);
});

test('an older native side without the method gets a clear error', async () => {
  installNativeMock();
  const device = await makeDevice();
  delete globalThis.NativeModules.WebGPU.importSharedTexture;

  await assert.rejects(() => importSharedTexture(device, { key: 'k', url: 'a.png' }), /needs updating/);
});
