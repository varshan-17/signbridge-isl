import type { Point } from './recognition';
export type PosePoint = Point & { visibility?: number };
export type WordFrame = number[][];
const bodyIndexes = [0,2,5,11,12,13,14];
const handIndexes = [0,4,5,8,9,12,13,16,17,20];

export function wordFrame(body: PosePoint[], hands: Point[][]): WordFrame | null {
  if (body.length !== 33 || [0,11,12].some(i=>(body[i].visibility ?? 1)<.5) || body.some(p=>!Number.isFinite(p.x)||!Number.isFinite(p.y))) return null;
  if (Math.hypot(body[11].x-body[12].x,body[11].y-body[12].y)<.08) return null;
  // Holistic uses anatomical left/right slots. Associate by pose wrists rather
  // than relying on the selfie-view handedness convention of the hand tracker.
  const distance = (hand: Point[], index: number)=>Math.hypot(hand[0].x-body[index].x,hand[0].y-body[index].y);
  const valid = hands.filter(hand=>hand.length===21 && hand.every(p=>Number.isFinite(p.x)&&Number.isFinite(p.y)));
  let left: Point[] | undefined, right: Point[] | undefined;
  if(valid.length>=2) {
    const straight=distance(valid[0],15)+distance(valid[1],16), crossed=distance(valid[1],15)+distance(valid[0],16);
    [left,right]=straight<=crossed ? [valid[0],valid[1]] : [valid[1],valid[0]];
  } else if(valid.length===1) { if(distance(valid[0],15)<=distance(valid[0],16)) left=valid[0]; else right=valid[0]; }
  return [...bodyIndexes.map(i=>[body[i].x,body[i].y]),...handIndexes.map(i=>left?[left[i].x,left[i].y]:[0,0]),...handIndexes.map(i=>right?[right[i].x,right[i].y]:[0,0])];
}

export function normalizeWordClip(frames: WordFrame[], count=48): Float32Array {
  if(frames.length<12 || frames.some(frame=>frame.length!==27 || frame.some(p=>p.length!==2 || p.some(v=>!Number.isFinite(v))))) throw new Error('Keep your face, shoulders and signing hands visible throughout the capture.');
  let centerX=0,centerY=0,scale=0;
  for(const frame of frames) {
    centerX+=(frame[3][0]+frame[4][0])/2; centerY+=(frame[3][1]+frame[4][1])/2;
    scale+=Math.hypot(frame[3][0]-frame[4][0],frame[3][1]-frame[4][1]);
  }
  centerX/=frames.length; centerY/=frames.length; scale/=frames.length;
  if(scale<.08) throw new Error('Move into view so both shoulders are visible.');
  const output=new Float32Array(2*count*27);
  for(let t=0;t<count;t++) {
    const frame=frames[Math.round(t*(frames.length-1)/(count-1))];
    for(let joint=0;joint<27;joint++) {
      output[t*27+joint]=(frame[joint][0]-centerX)/scale;
      output[count*27+t*27+joint]=(frame[joint][1]-centerY)/scale;
    }
  }
  return output;
}
