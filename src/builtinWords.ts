import * as ort from 'onnxruntime-web/wasm';
import wasmUrl from '../node_modules/onnxruntime-web/dist/ort-wasm-simd-threaded.wasm?url';
import wasmModuleUrl from '../node_modules/onnxruntime-web/dist/ort-wasm-simd-threaded.mjs?url';
import { normalizeWordClip } from './wordFeatures';
import type { WordFrame } from './wordFeatures';
import { softmax } from './builtinLetters';
import { selectWordConsensus } from './recognitionStability.ts';

export type WordMetadata = {classes:string[];shape:number[]};
export class WordRecognizer {
  private session: ort.InferenceSession;
  public metadata: WordMetadata;
  constructor(session: ort.InferenceSession, metadata:WordMetadata) {this.session=session;this.metadata=metadata;}
  async predict(frames:WordFrame[]): Promise<{label:string;score:number;kind:'word';id:string} | null> {
    // Compare the full capture with its middle 80%, reducing dependence on
    // settling into a pose or lowering the hands at the capture boundaries.
    const trim=Math.min(Math.floor(frames.length*.1),Math.floor((frames.length-12)/2));
    const views=[frames,trim>0?frames.slice(trim,-trim):frames];
    const probabilities:number[][]=[];
    for(const view of views) {
      const tensor=new ort.Tensor('float32',normalizeWordClip(view,this.metadata.shape[2]),this.metadata.shape);
      try {
        const outputs=await this.session.run({poses:tensor});
        try { probabilities.push(softmax(outputs.logits.data as Float32Array)); }
        finally { Object.values(outputs).forEach(output=>output.dispose()); }
      } finally { tensor.dispose(); }
    }
    return selectWordConsensus(this.metadata.classes,probabilities);
  }
}

let wordModelPromise:Promise<WordRecognizer> | null = null;
export function loadWordRecognizer():Promise<WordRecognizer> {
  if(!wordModelPromise) wordModelPromise=(async()=>{
    ort.env.wasm.numThreads=1;
    ort.env.wasm.wasmPaths={wasm:wasmUrl,mjs:wasmModuleUrl};
    const response=await fetch('/models/isl-words.json');
    if(!response.ok) throw new Error('The built-in word model could not load. Check the app’s model files and refresh.');
    const metadata=await response.json() as WordMetadata;
    if(metadata.classes.length!==263 || JSON.stringify(metadata.shape)!=='[1,2,120,27]') throw new Error('Invalid ISL word model.');
    const session=await ort.InferenceSession.create('/models/isl-words.onnx',{executionProviders:['wasm'],graphOptimizationLevel:'all'});
    return new WordRecognizer(session,metadata);
  })().catch(e=>{wordModelPromise=null;throw e;});
  return wordModelPromise;
}
