import type {Point} from './recognition';

export type LetterExample={label:string;features:number[];source:string};
/** Preserve relative hand position and orientation: these distinguish two-hand
 * letters that an independently rotated hand descriptor can collapse. */
export function referenceFeatures(hands:Point[][],width:number,height:number):number[]|null {
  if(!width||!height) return null;
  const size=Math.max(width,height);
  const valid=hands.filter(hand=>hand.length===21 && hand.every(p=>Number.isFinite(p.x)&&Number.isFinite(p.y)&&Number.isFinite(p.z))
    && Math.hypot((hand[9].x-hand[0].x)*width,(hand[9].y-hand[0].y)*height,(hand[9].z-hand[0].z)*width)>=size*.035);
  if(!valid.length) return null;
  const spatial=valid.slice(0,2).sort((a,b)=>a[0].x-b[0].x).map(hand=>hand.map(p=>({x:p.x*width/size,y:p.y*height/size,z:p.z*width/size})));
  const scale=spatial.reduce((sum,h)=>sum+Math.hypot(h[9].x-h[0].x,h[9].y-h[0].y,h[9].z-h[0].z),0)/spatial.length;
  const slots=[spatial[0],spatial[1]];
  return [...slots.flatMap(hand=>[hand?2:0,...Array.from({length:21},(_,i)=>hand?[(hand[i].x-hand[0].x)/scale,(hand[i].y-hand[0].y)/scale,(hand[i].z-hand[0].z)/scale*.35]:[0,0,0]).flat()]),...(slots[1]?[(slots[1][0].x-slots[0][0].x)/scale,(slots[1][0].y-slots[0][0].y)/scale,0]:[0,0,0])];
}
export function referenceDistance(a:number[],b:number[]):number {
  if(a.length!==131||b.length!==131||a[0]!==b[0]||a[64]!==b[64]) return Infinity;
  return Math.sqrt(a.reduce((sum,v,i)=>sum+(v-b[i])**2,0)/131);
}
export function rankReferenceLetters(features:number[],examples:LetterExample[]) {
  const groups=new Map<string,number[]>();
  for(const example of examples) {
    const distances=groups.get(example.label)??[];
    distances.push(referenceDistance(features,example.features));groups.set(example.label,distances);
  }
  return Array.from(groups,([label,distances])=>{
    distances.sort((a,b)=>a-b);
    // Requiring two supporting examples avoids accepting one tracking accident.
    const compatible=distances.filter(Number.isFinite).length;
    const support=distances.filter(distance=>distance<=.24).length;
    return {label,distance:compatible>=2?(distances[0]+distances[1])/2:distances[0],support};
  }).sort((a,b)=>a.distance-b.distance);
}
