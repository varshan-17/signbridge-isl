import {test} from 'node:test';
import assert from 'node:assert/strict';
import {handFrameQuality,PredictionStabilizer,selectWordConsensus,wordCaptureIsUsable} from '../src/recognitionStability.ts';
import {rankReferenceLetters} from '../src/referenceLetters.ts';
import {CommitGate,appendSign} from '../src/recognition.ts';

const a={id:'letter-A',label:'A',kind:'letter',score:90};
const b={...a,id:'letter-B',label:'B'};
test('letter consensus rejects flicker, stale observations and unknown gaps',()=>{
  const filter=new PredictionStabilizer();
  [a,b,a,b,a,b].forEach((p,i)=>assert.equal(filter.update(p,i*120),null));
  filter.reset();
  [0,120,240].forEach(t=>assert.equal(filter.update(a,t),null));
  assert.equal(filter.update(a,360).label,'A');
  assert.equal(filter.update(null,480),null);
  assert.equal(filter.update(a,600),null);
  assert.equal(filter.update(a,1100),null);
  assert.equal(filter.update(a,1100),null);
});
test('a brief competing label cannot immediately replace a stable letter',()=>{
  const filter=new PredictionStabilizer();
  [a,a,b,a].forEach((p,i)=>assert.equal(filter.update(p,i*120),null));
  assert.equal(filter.update(a,480).label,'A');
  assert.equal(filter.update(b,600),null);
});
test('stabilized letters commit once, then can repeat after hand release',()=>{
  const filter=new PredictionStabilizer(),gate=new CommitGate();
  let text='';
  for(let now=0;now<=2400;now+=120) {
    const p=filter.update(a,now);
    if(gate.update(p?.id??null,true,now).commit)text=appendSign(text,p);
  }
  assert.equal(text,'A');
  filter.reset(); gate.update(null,false,2520);gate.update(null,false,3000);
  for(let now=3120;now<=5520;now+=120) {
    const p=filter.update(a,now);
    if(gate.update(p?.id??null,true,now).commit)text=appendSign(text,p);
  }
  assert.equal(text,'AA');
});
test('letter quality rejects clipped fingers and a small second hand',()=>{
  const hand=Array.from({length:21},(_,i)=>({x:.5,y:.5-i*.008,z:0}));
  assert.equal(handFrameQuality([hand],960,720),'ready');
  const clipped=hand.map(p=>({...p}));clipped[8].x=1.05;
  assert.equal(handFrameQuality([clipped],960,720),'out-of-frame');
  const small=hand.map(p=>({...p,y:.5+(p.y-.5)*.1}));
  assert.equal(handFrameQuality([hand,small],960,720),'too-small');
  assert.equal(handFrameQuality([],960,720),'no-hands');
});
test('reference support counts close examples rather than every compatible hand count',()=>{
  const feature=Array(131).fill(0);feature[0]=2;
  const far=feature.map((v,i)=>i===0||i===64?v:v+.4);
  const rank=rankReferenceLetters(feature,[{label:'A',features:feature,source:'near'},{label:'A',features:far,source:'far'}]);
  assert.equal(rank[0].support,1);
  assert.ok(rank[0].distance<.24);
});
test('word views must agree, preserve duplicate gloss mass and reject ambiguous scores',()=>{
  const labels=['HELLO','HELLO','THANK YOU'];
  assert.equal(selectWordConsensus(labels,[[.45,.4,.15],[.4,.4,.2]]).label,'HELLO');
  assert.equal(selectWordConsensus(labels,[[.45,.4,.15],[.1,.1,.8]]),null);
  assert.equal(selectWordConsensus(['HELLO','THANK YOU'],[[.6,.4],[.6,.4]]),null);
  assert.equal(selectWordConsensus(labels,[[NaN,.4,.15],[.4,.4,.2]]),null);
  assert.equal(selectWordConsensus(labels,[[.45,.4,.15]]),null);
});
test('word capture rejects long tracking gaps even with enough total valid frames',()=>{
  const times=Array.from({length:40},(_,i)=>50+i*100);
  assert.equal(wordCaptureIsUsable(40,30,times,0,4000),true);
  assert.equal(wordCaptureIsUsable(40,30,times.filter((_,i)=>i<10||i>=18),0,4000),false);
  assert.equal(wordCaptureIsUsable(40,10,times,0,4000),false);
  assert.equal(wordCaptureIsUsable(40,30,times.slice(0,20),0,4000),false);
});
