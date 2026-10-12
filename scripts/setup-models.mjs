import { mkdir, writeFile, copyFile } from 'node:fs/promises';
import { createHash } from 'node:crypto';

await mkdir('public/models',{recursive:true});
await mkdir('public/vision',{recursive:true});
const revision = '62c07b470c5a67dcadda6e7429a76e6a1766b749';
const source = `https://huggingface.co/venmugilrajan/SignLens-Landmark-Model/resolve/${revision}/isl_landmarks_weights.json`;
const response = await fetch(source,{signal:AbortSignal.timeout(60000)});
if(!response.ok) throw new Error(`ISL alphabet model download failed: ${response.status}`);
const bytes = new Uint8Array(await response.arrayBuffer());
const model = JSON.parse(new TextDecoder().decode(bytes));
const shapes = [[156,512],[512,256],[256,128],[128,35]];
const data = [];
for(let i=0;i<4;i++) {
 const weights=model[`w${i}`],bias=model[`b${i}`];
 if(weights.length!==shapes[i][0] || weights.some(row=>row.length!==shapes[i][1]) || bias.length!==shapes[i][1]) throw new Error('Unexpected model shape.');
 for(const row of weights) for(const value of row) data.push(value);
 for(const value of bias) data.push(value);
}
const floats = new Float32Array(data);
await writeFile('public/models/isl-letters.bin',new Uint8Array(floats.buffer));
await writeFile('public/models/isl-letters.json',JSON.stringify({classes:model.classes,shapes,source,revision,sourceSha256:createHash('sha256').update(bytes).digest('hex')},null,2));
await copyFile('licenses/Apache-2.0.txt','public/models/LICENSE.txt');
await copyFile('MODEL_NOTICES.md','public/models/NOTICE.md');

// Vite bundles the ONNX runtime assets from the installed package.
const poseUrl = 'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task';
const pose = await fetch(poseUrl,{signal:AbortSignal.timeout(60000)});
if(!pose.ok) throw new Error(`Body tracker download failed: ${pose.status}`);
await writeFile('public/vision/pose_landmarker.task',new Uint8Array(await pose.arrayBuffer()));
console.log('Built-in alphabet weights, word runtime assets and body tracker ready.');
