import * as ort from 'onnxruntime-web/wasm';
import wasmUrl from '../node_modules/onnxruntime-web/dist/ort-wasm-simd-threaded.wasm?url';
import wasmModuleUrl from '../node_modules/onnxruntime-web/dist/ort-wasm-simd-threaded.mjs?url';
import { normalizeWordClip } from './wordFeatures';
import type { WordFrame } from './wordFeatures';
import { softmax } from './builtinLetters';

export type WordMetadata = {classes:string[];shape:number[]};
export class WordRecognizer {
  private session: ort.InferenceSession;
  public metadata: WordMetadata;
  constructor(session: ort.InferenceSession, metadata:WordMetadata) {this.session=session;this.metadata=metadata;}
  async predict(frames:WordFrame[]): Promise<{label:string;score:number;kind:'word';id:string} | null> {
    const tensor=new ort.Tensor('float32',normalizeWordClip(frames,this.metadata.shape[2]),this.metadata.shape);
    try {
      const outputs=await this.session.run({poses:tensor});
      const values=outputs.logits.data as Float32Array;
      const grouped=new Map<string,number>();
      softmax(values).forEach((score,index)=>{const label=this.metadata.classes[index];grouped.set(label,(grouped.get(label)??0)+score);});
      const scores=Array.from(grouped,([label,score])=>({label,score})).sort((a,b)=>b.score-a.score);
      Object.values(outputs).forEach(output=>output.dispose());
      if(scores[0].score<.7 || scores[0].score-scores[1].score<.25) return null;
      return {label:scores[0].label,score:Math.round(scores[0].score*100),kind:'word',id:`builtin-word-${scores[0].label}`};
    } finally { tensor.dispose(); }
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
