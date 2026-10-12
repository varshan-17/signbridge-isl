import type { Point } from './recognition';

export type Prediction = { id: string; label: string; kind: 'letter' | 'word'; score: number; scoreType?: 'match' };

export function handFrameQuality(hands: Point[][], width: number, height: number): 'ready' | 'no-hands' | 'out-of-frame' | 'too-small' {
  if (!hands.length) return 'no-hands';
  if (width <= 0 || height <= 0 || hands.some(hand => hand.length !== 21 || hand.some(p => !Number.isFinite(p.x) || !Number.isFinite(p.y) || !Number.isFinite(p.z) || p.x < 0 || p.x > 1 || p.y < 0 || p.y > 1))) return 'out-of-frame';
  if (hands.some(hand => Math.hypot((hand[9].x-hand[0].x)*width, (hand[9].y-hand[0].y)*height) < 25)) return 'too-small';
  return 'ready';
}

/** A new letter needs recent, repeated evidence. Unknown frames and stale
 * observations clear evidence rather than extending an old confident result. */
export class PredictionStabilizer {
  private history: { prediction: Prediction; time: number }[] = [];
  reset() { this.history = []; }
  update(prediction: Prediction | null | undefined, now: number): Prediction | null {
    const last = this.history.at(-1);
    if (!prediction || !Number.isFinite(prediction.score) || !Number.isFinite(now)) { this.reset(); return null; }
    if (last && (now-last.time > 350 || now <= last.time)) this.reset();
    this.history.push({ prediction, time: now });
    this.history = this.history.filter(item => now-item.time <= 700).slice(-5);
    const agreeing = this.history.filter(item => item.prediction.id === prediction.id);
    if (agreeing.length < 4 || agreeing.length/this.history.length < .8 || now-agreeing[0].time < 320) return null;
    return { ...prediction, score: Math.round(agreeing.reduce((sum,item) => sum+item.prediction.score,0)/agreeing.length) };
  }
}

/** Group duplicate gloss labels before comparing two temporal views. A single
 * high score is not enough when the cropped view prefers a different word. */
export function selectWordConsensus(labels: string[], passes: number[][]): (Prediction & {kind:'word'}) | null {
  if (passes.length < 2 || passes.some(pass => pass.length !== labels.length || pass.some(p => !Number.isFinite(p) || p < 0))) return null;
  const ranks = passes.map(pass => {
    const groups = new Map<string, number>();
    pass.forEach((p,i) => groups.set(labels[i],(groups.get(labels[i]) ?? 0)+p));
    return Array.from(groups,([label,score]) => ({label,score})).sort((a,b) => b.score-a.score);
  });
  const label = ranks[0][0]?.label;
  if (!label || ranks.some(rank => rank[0]?.label !== label || rank[0].score < .6 || rank[0].score-(rank[1]?.score ?? 0) < .15)) return null;
  const average = new Map<string,number>();
  ranks.forEach(rank => rank.forEach(item => average.set(item.label,(average.get(item.label) ?? 0)+item.score/ranks.length)));
  const combined = Array.from(average,([label,score]) => ({label,score})).sort((a,b) => b.score-a.score);
  if (combined[0].score < .7 || combined[0].score-(combined[1]?.score ?? 0) < .25) return null;
  return {id:`builtin-word-${label}`,label,score:Math.round(combined[0].score*100),kind:'word'};
}

export function wordCaptureIsUsable(total: number, handFrames: number, times: number[], start: number, end: number): boolean {
  if (total < 16 || times.length < 16 || times.length/total < .75 || handFrames/total < .6 || end <= start) return false;
  const boundaries = [start,...times,end];
  return boundaries.every((time,i) => Number.isFinite(time) && (i === 0 || time >= boundaries[i-1] && time-boundaries[i-1] <= 500));
}
