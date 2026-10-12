import { test } from 'node:test';
import assert from 'node:assert/strict';
import { appendSign, CommitGate, extractFeatures, FEATURE_SIZE, prepareSequence, recognize, resample, sequenceDistance, validateLibrary } from '../src/recognition.ts';

function hand(offset = 0, scale = 1) {
  return Array.from({length:21}, (_,i) => ({x:offset + i * 0.012 * scale, y:offset + (i % 5) * 0.02 * scale, z:i * 0.001 * scale}));
}
const frame = extractFeatures([hand()], ['Left']);
const sequence = resample([frame]);
const prepared = prepareSequence(sequence);
const sign = {id:'a',label:'A',kind:'letter',samples:[prepared,prepared,prepared]};

test('two-hand landmarks retain slot identity, hand count and relative position', () => {
  assert.equal(frame.length, FEATURE_SIZE);
  const two = extractFeatures([hand(),hand(.4)], ['Left','Right']);
  assert.equal(two[0],2); assert.equal(two[64],2); assert.notEqual(two[128],0);
  assert.deepEqual(two,extractFeatures([hand(.4),hand()],['Right','Left']));
  assert.equal(extractFeatures([],[]),null);
});
test('features are invariant to image translation and uniform scale', () => {
  const moved = extractFeatures([hand(.2,2)],['Left']);
  assert.ok(sequenceDistance(prepareSequence([moved]),prepared) < 0.00001);
});
test('temporal classifier matches taught signs and abstains on unknown/ambiguous poses', () => {
  assert.equal(recognize(sequence,[sign]).sign.label,'A');
  assert.equal(recognize(sequence,[{...sign,samples:[sequence,sequence]}]),null);
  assert.equal(recognize(sequence,[sign,{...sign,id:'b',label:'B'}]),null);
  const different = resample([frame.map((v,i) => i===0||i===64?v:v+3)]);
  assert.equal(recognize(different,[sign]),null);
  const wrongHand = resample([extractFeatures([hand()],['Right'])]);
  assert.equal(recognize(wrongHand,[sign]),null);
});
test('DTW recognizes movement at a different speed and distinguishes different paths', () => {
  const shape = (t) => frame.map((v,i)=>i===0||i===64?v:v + Math.sin(t*Math.PI)*.12);
  const slow = resample(Array.from({length:30},(_,i)=>shape(i/29)));
  const fast = resample(Array.from({length:18},(_,i)=>shape((i/17)**.8)));
  assert.ok(sequenceDistance(slow,fast) < .04);
});
test('whole-hand trajectories are retained independently of signing position', () => {
  const moving = Array.from({length:12},(_,i)=>extractFeatures([hand(i*.015)],['Left']));
  const moved = Array.from({length:12},(_,i)=>extractFeatures([hand(.3+i*.015)],['Left']));
  assert.ok(sequenceDistance(prepareSequence(moving),prepareSequence(moved)) < 0.00001);
  assert.ok(sequenceDistance(prepareSequence(moving),prepared) > .05);
});
test('commit gate waits, prevents duplicate letters, and allows repeated L after release', () => {
  const gate = new CommitGate();
  assert.equal(gate.update('L',true,0).commit,false);
  assert.equal(gate.update('L',true,700).commit,false);
  assert.equal(gate.update('L',true,950).commit,true);
  assert.equal(gate.update('L',true,2000).commit,false);
  gate.update(null,false,2100); gate.update(null,false,2500);
  gate.update('L',true,2600);
  assert.equal(gate.update('L',true,3600).commit,true);
  gate.update('O',true,3700);
  assert.equal(gate.update('O',true,4700).commit,true);
});
test('uncertain frames do not rearm a previously committed sign', () => {
  const gate = new CommitGate(); gate.update('A',true,0); gate.update('A',true,1000);
  gate.update(null,true,1100); gate.update('A',true,1200);
  assert.equal(gate.update('A',true,3000).commit,false);
});
test('letters build words; word signs keep boundaries', () => {
  assert.equal(appendSign('HEL',{kind:'letter',label:'L'}),'HELL');
  assert.equal(appendSign('HELLO ',{kind:'word',label:'WORLD'}),'HELLO WORLD ');
  assert.equal(appendSign('',{kind:'word',label:'THANK YOU'}),'THANK YOU ');
});
test('library import validates shape and rejects malformed, oversized or duplicate data', () => {
  assert.deepEqual(validateLibrary({version:1,signs:[sign]}).signs[0],sign);
  assert.throws(()=>validateLibrary({version:2,signs:[]}));
  assert.throws(()=>validateLibrary({version:1,signs:[sign,sign]}));
  assert.throws(()=>validateLibrary({version:1,signs:[{...sign,label:'<script>'}]}));
  assert.throws(()=>validateLibrary({version:1,signs:[{...sign,samples:[[[Infinity]]]}]}));
  assert.throws(()=>validateLibrary({version:1,signs:[{...sign,samples:[Array.from({length:12},()=>Array(FEATURE_SIZE).fill(0))]}]}));
});
