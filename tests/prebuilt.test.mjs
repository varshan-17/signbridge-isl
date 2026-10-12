import {test} from 'node:test';
import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import {LetterRecognizer, letterFeatures, softmax} from '../src/builtinLetters.ts';
import {wordFrame, normalizeWordClip} from '../src/wordFeatures.ts';
import {handFrameQuality} from '../src/recognitionStability.ts';

const hand = (x=.25,y=.5)=>Array.from({length:21},(_,i)=>({x:x+i*.008,y:y-(i%5)*.025,z:-i*.001}));
test('built-in alphabet preprocessing uses spatial slots and exact 156 features',()=>{
  const left=hand(.2),right=hand(.6);
  const features=letterFeatures([right,left],960,720);
  assert.equal(features.length,156);
  assert.deepEqual(features,letterFeatures([left,right],960,720));
  assert.ok(features.every(Number.isFinite));
  assert.ok(Math.abs(features[27])<1e-8);
  assert.ok(Math.abs(features[28]+1)<1e-8);
  assert.ok(letterFeatures([left],960,720).slice(78).every(v=>v===0));
  assert.equal(letterFeatures([],960,720),null);
  assert.equal(letterFeatures([left.map(()=>({x:.1,y:.1,z:0}))],960,720),null);
});
test('bundled alphabet weights load and reject absent hands and model corruption',async()=>{
  const metadata=JSON.parse(await readFile('public/models/isl-letters.json','utf8'));
  const bytes=await readFile('public/models/isl-letters.bin');
  const model=new LetterRecognizer(metadata,bytes.buffer.slice(bytes.byteOffset,bytes.byteOffset+bytes.byteLength));
  assert.equal(model.predict([],960,720),null);
  const logits=model.logits(letterFeatures([hand()],960,720));
  assert.equal(logits.length,35);
  assert.ok(logits.every(Number.isFinite));
  assert.ok(Math.abs(softmax(logits).reduce((a,b)=>a+b,0)-1)<1e-8);
  assert.throws(()=>new LetterRecognizer(metadata,new ArrayBuffer(16)));
});
test('digit and uncertain outputs are not promoted into confident letter predictions',async()=>{
  const metadata=JSON.parse(await readFile('public/models/isl-letters.json','utf8'));
  const count=metadata.shapes.reduce((sum,[i,o])=>sum+i*o+o,0);
  const weights=new Float32Array(count);
  let model=new LetterRecognizer(metadata,weights.buffer);
  assert.equal(model.predict([hand()],960,720),null);
  weights[count-35+metadata.classes.indexOf('1')]=20;
  model=new LetterRecognizer(metadata,weights.buffer);
  assert.equal(model.predict([hand()],960,720),null);
});
test('reference-photo regressions do not insert the wrong letter or accept cropped hands',async()=>{
  const metadata=JSON.parse(await readFile('public/models/isl-letters.json','utf8'));
  const bytes=await readFile('public/models/isl-letters.bin');
  const examples=JSON.parse(await readFile('public/models/isl-letter-examples.json','utf8')).examples;
  const model=new LetterRecognizer(metadata,bytes.buffer.slice(bytes.byteOffset,bytes.byteOffset+bytes.byteLength),examples);
  const fixtures=JSON.parse(await readFile('tests/fixtures/letter-photos.json','utf8'));
  for(const fixture of fixtures) {
    const prediction=model.predict(fixture.hands,fixture.width,fixture.height);
    if(handFrameQuality(fixture.hands,fixture.width,fixture.height)!=='ready')assert.equal(prediction,null);
    else if(prediction)assert.equal(prediction.label,fixture.label);
  }
  // These source-derived references are regression fixtures, not a held-out
  // real-world accuracy benchmark.
  assert.equal(model.predict(fixtures[0].hands,fixtures[0].width,fixtures[0].height)?.label,'A');
});
function body(){
  const result=Array.from({length:33},()=>({x:.5,y:.4,z:0,visibility:1}));
  result[11].x=.35;result[12].x=.65;result[15].x=.2;result[16].x=.7;
  return result;
}
test('word features preserve anatomical slots and whole-clip shoulder normalization',()=>{
  const pose=body(),left=hand(.2),right=hand(.7);
  const frame=wordFrame(pose,[right,left]);
  assert.equal(frame.length,27);
  assert.deepEqual(frame,wordFrame(pose,[left,right]));
  assert.equal(frame[7][0],left[0].x);
  assert.equal(frame[17][0],right[0].x);
  const clip=Array.from({length:24},()=>frame);
  const output=normalizeWordClip(clip);
  assert.equal(output.length,2*48*27);
  assert.ok(Math.abs(output[3]+.5)<1e-6);
  assert.ok(Math.abs(output[4]-.5)<1e-6);
  assert.equal(wordFrame([],[]),null);
  pose[11].visibility=.1;assert.equal(wordFrame(pose,[left]),null);
  assert.throws(()=>normalizeWordClip([]));
});
test('converted word network matches the original PyTorch forward pass',async()=>{
  const ort=await import('onnxruntime-web/wasm');
  ort.env.wasm.numThreads=1;
  const session=await ort.InferenceSession.create(await readFile('public/models/isl-words.onnx'),{executionProviders:['wasm']});
  const fixture=JSON.parse(await readFile('tests/fixtures/word-parity.json','utf8'));
  const input=new ort.Tensor('float32',Float32Array.from(fixture.input),fixture.shape);
  const output=await session.run({poses:input});
  assert.equal(output.logits.data.length,263);
  const errors=Array.from(output.logits.data,(value,i)=>Math.abs(value-fixture.logits[i]));
  assert.ok(Math.max(...errors)<.0001,`Conversion error ${Math.max(...errors)}`);
  const meta=JSON.parse(await readFile('public/models/isl-words.json','utf8'));
  assert.equal(meta.classes.length,263);
  assert.deepEqual(meta.originalClasses,[...meta.originalClasses].sort());
  assert.ok(meta.classes.includes('HELLO'));assert.ok(meta.classes.includes('THANK YOU'));
  input.dispose();output.logits.dispose();await session.release();
});
