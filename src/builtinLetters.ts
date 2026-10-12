import type { Point } from './recognition';
import {rankReferenceLetters,referenceFeatures} from './referenceLetters.ts';
import type {LetterExample} from './referenceLetters';

export const ISL_LETTERS = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'.split('');
const fingerJoints = [[0,1,2],[1,2,3],[2,3,4],[0,5,6],[5,6,7],[6,7,8],[0,9,10],[9,10,11],[10,11,12],[0,13,14],[13,14,15],[14,15,16],[0,17,18],[17,18,19],[18,19,20]];
const oneHandLetters = new Set(['C','I','J','L','O','Q','U','V']);

/** Matches SignLens's published ISL preprocessing: pixel aspect correction,
 * wrist centering, palm scaling, its trained rotation convention and joint angles.
 * Slots use raw wrist-x ordering, not handedness or the mirrored display. */
export function letterFeatures(hands: Point[][], width: number, height: number): number[] | null {
  if (width <= 0 || height <= 0) return null;
  const valid = hands.filter(hand => hand.length === 21 && hand.every(p => Number.isFinite(p.x) && Number.isFinite(p.y) && Number.isFinite(p.z) && p.x >= -.2 && p.x <= 1.2 && p.y >= -.2 && p.y <= 1.2)
    && Math.hypot((hand[9].x-hand[0].x)*width,(hand[9].y-hand[0].y)*height) >= 25);
  if (!valid.length) return null;
  const slots = valid.slice(0,2).sort((a,b) => a[0].x-b[0].x).map(hand => {
    const points = hand.map(p => [p.x*width,p.y*height,p.z*width]);
    const wrist = points[0];
    const centered = points.map(p => p.map((v,i) => v-wrist[i]));
    const scale = Math.hypot(centered[9][0],centered[9][1]);
    const angle = Math.atan2(centered[9][0]/scale,-centered[9][1]/scale);
    const cos = Math.cos(angle), sin = Math.sin(angle);
    const result = centered.flatMap(p => [(p[0]*cos+p[1]*sin)/scale,(-p[0]*sin+p[1]*cos)/scale,p[2]/scale]);
    for(const [a,b,c] of fingerJoints) {
      const ab = points[a].map((v,i)=>v-points[b][i]), cb = points[c].map((v,i)=>v-points[b][i]);
      const dot = ab.reduce((s,v,i)=>s+v*cb[i],0);
      const denominator = Math.hypot(...ab)*Math.hypot(...cb)+1e-6;
      result.push(Math.acos(Math.max(-1,Math.min(1,dot/denominator)))/Math.PI);
    }
    return result;
  });
  return [...slots[0],...(slots[1] ?? Array(78).fill(0))];
}

export function softmax(logits: ArrayLike<number>): number[] {
  const values = Array.from(logits), max = Math.max(...values);
  const exps = values.map(x=>Math.exp(x-max));
  const sum = exps.reduce((a,b)=>a+b,0);
  return exps.map(v=>v/sum);
}

export type LetterModelMetadata = { classes: string[]; shapes: [number,number][] };
export class LetterRecognizer {
  public metadata: LetterModelMetadata;
  private layers: { weights: Float32Array; bias: Float32Array; inputs: number; outputs: number }[];
  private examples: LetterExample[];
  constructor(metadata: LetterModelMetadata, data: ArrayBuffer, examples:LetterExample[]=[]) {
    this.examples=examples;
    if(examples.some(example=>!ISL_LETTERS.includes(example.label)||example.features.length!==131||example.features.some(v=>!Number.isFinite(v)))) throw new Error('Invalid alphabet reference examples.');
    this.metadata = metadata;
    if (!Array.isArray(metadata.classes) || metadata.classes.length !== 35 || ISL_LETTERS.some(letter=>!metadata.classes.includes(letter))) throw new Error('Invalid ISL alphabet model.');
    const expected = [[156,512],[512,256],[256,128],[128,35]];
    if (JSON.stringify(metadata.shapes) !== JSON.stringify(expected)) throw new Error('Invalid alphabet model dimensions.');
    const values = new Float32Array(data);
    if (values.length !== expected.reduce((s,[i,o])=>s+i*o+o,0) || values.some(v=>!Number.isFinite(v))) throw new Error('Invalid alphabet weights.');
    let offset = 0;
    this.layers = expected.map(([inputs,outputs]) => {
      const weights = values.subarray(offset,offset+inputs*outputs); offset += inputs*outputs;
      const bias = values.subarray(offset,offset+outputs); offset += outputs;
      return {weights,bias,inputs,outputs};
    });
  }
  logits(features: number[]): Float32Array {
    if(features.length !== 156 || features.some(v=>!Number.isFinite(v))) throw new Error('Invalid alphabet features.');
    let input: ArrayLike<number> = features;
    for (let index=0;index<this.layers.length;index++) {
      const {weights,bias,inputs,outputs} = this.layers[index];
      const out = new Float32Array(bias);
      for(let i=0;i<inputs;i++) {
        const x=input[i]; if(x===0) continue;
        for(let j=0;j<outputs;j++) out[j] += x*weights[i*outputs+j];
      }
      if(index<this.layers.length-1) for(let j=0;j<outputs;j++) out[j]=Math.max(0,out[j]);
      input=out;
    }
    return input as Float32Array;
  }
  predict(hands: Point[][], width: number, height: number): {label:string;score:number;kind:'letter';id:string;scoreType?:'match'} | null {
    const features = letterFeatures(hands,width,height);
    if(!features) return null;
    const probabilities = softmax(this.logits(features));
    const oneHand = features.slice(78).every(v=>v===0);
    // Keep the full 35-way softmax: masking digits must not inflate letter scores.
    const ranks = probabilities.map((score,i)=>({score,label:this.metadata.classes[i]})).sort((a,b)=>b.score-a.score);
    const best = ranks[0];
    if(this.examples.length) {
      const spatial=referenceFeatures(hands,width,height);
      if(!spatial) return null;
      const candidates=rankReferenceLetters(spatial,this.examples), match=candidates[0];
      if(!match || match.distance>.24 || candidates[1].distance-match.distance<.035) return null;
      // A single compatible example needs independent agreement from the MLP.
      if(match.support<2 && (best.label!==match.label || best.score<.8)) return null;
      return {label:match.label,score:Math.round(Math.exp(-match.distance*4)*100),kind:'letter',id:`builtin-letter-${match.label}`,scoreType:'match'};
    }
    if (!ISL_LETTERS.includes(best.label) || best.score < .8 || best.score-ranks[1].score < .2) return null;
    if(oneHand !== oneHandLetters.has(best.label)) return null;
    return {label:best.label,score:Math.round(best.score*100),kind:'letter',id:`builtin-letter-${best.label}`};
  }
}

let modelPromise: Promise<LetterRecognizer> | null = null;
export function loadLetterRecognizer(): Promise<LetterRecognizer> {
  if(!modelPromise) modelPromise = Promise.all([fetch('/models/isl-letters.json'),fetch('/models/isl-letters.bin'),fetch('/models/isl-letter-examples.json')]).then(async ([meta,weights,references])=>{
    if(!meta.ok || !weights.ok || !references.ok) throw new Error('The built-in alphabet model is missing. Run npm run setup:models.');
    return new LetterRecognizer(await meta.json(),await weights.arrayBuffer(),(await references.json()).examples);
  }).catch(error=>{modelPromise=null;throw error;});
  return modelPromise;
}
