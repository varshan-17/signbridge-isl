import { access, mkdir, copyFile, readdir, writeFile } from 'node:fs/promises';
import { resolve } from 'node:path';

const destination = resolve('public/vision');
await mkdir(destination, { recursive: true });
const wasmSource = resolve('node_modules/@mediapipe/tasks-vision/wasm');
for (const name of await readdir(wasmSource)) {
  if (/\.(wasm|js)$/.test(name)) await copyFile(resolve(wasmSource, name), resolve(destination, name));
}
async function ensureModel(filename, url) {
  const path = resolve(destination, filename);
  try {
    await access(path);
    return;
  } catch {
    const response = await fetch(url, { signal: AbortSignal.timeout(60000) });
    if (!response.ok) throw new Error(`${filename} download failed: ${response.status}`);
    await writeFile(path, new Uint8Array(await response.arrayBuffer()));
  }
}
await ensureModel('hand_landmarker.task', 'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task');
await ensureModel('pose_landmarker.task', 'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task');
console.log('Local hand and pose trackers and WASM ready in public/vision.');
