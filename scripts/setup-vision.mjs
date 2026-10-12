import { access, mkdir, copyFile, readdir, writeFile } from 'node:fs/promises';
import { resolve } from 'node:path';

const destination = resolve('public/vision');
await mkdir(destination, { recursive: true });
const wasmSource = resolve('node_modules/@mediapipe/tasks-vision/wasm');
for (const name of await readdir(wasmSource)) {
  if (/\.(wasm|js)$/.test(name)) await copyFile(resolve(wasmSource, name), resolve(destination, name));
}
const url = 'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task';
const handModel = resolve(destination, 'hand_landmarker.task');
try {
  await access(handModel);
} catch {
  const response = await fetch(url, { signal: AbortSignal.timeout(60000) });
  if (!response.ok) throw new Error(`Hand tracker download failed: ${response.status}`);
  await writeFile(handModel, new Uint8Array(await response.arrayBuffer()));
}
console.log('Local hand tracker and WASM ready in public/vision.');
