export type Point = { x: number; y: number; z: number };
export type Frame = number[];
export type Sequence = Frame[];
export type Sign = { id: string; label: string; kind: 'letter' | 'word'; samples: Sequence[] };
export type Library = { version: 1; signs: Sign[] };
export const FEATURE_SIZE = 134;
export const FRAME_COUNT = 12;
export const MAX_SAMPLES = 8;
export const STORAGE_KEY = 'signbridge-library-v1';

// Handedness slots stay stable when the hands cross. Global wrist separation
// preserves two-hand relationships that independent normalization would erase.
export function extractFeatures(hands: Point[][], handedness: string[]): Frame | null {
  if (!hands.length) return null;
  const slots: (Point[] | undefined)[] = [undefined, undefined];
  for (let i = 0; i < hands.length; i++) {
    if (hands[i].length !== 21) continue;
    const preferred = handedness[i] === 'Left' ? 0 : 1;
    if (!slots[preferred]) slots[preferred] = hands[i];
    else if (!slots[1 - preferred]) slots[1 - preferred] = hands[i];
  }
  const present = slots.filter((h): h is Point[] => !!h);
  if (!present.length) return null;
  const scale = Math.max(0.02, present.reduce((s, h) => s + Math.hypot(h[9].x - h[0].x, h[9].y - h[0].y), 0) / present.length);
  const features: number[] = [];
  for (const hand of slots) {
    features.push(hand ? 2 : 0);
    for (let i = 0; i < 21; i++) {
      features.push(hand ? (hand[i].x - hand[0].x) / scale : 0);
      features.push(hand ? (hand[i].y - hand[0].y) / scale : 0);
      features.push(hand ? (hand[i].z - hand[0].z) / scale * 0.35 : 0);
    }
  }
  features.push(...(slots[0] && slots[1]
    ? [(slots[1][0].x - slots[0][0].x) / scale, (slots[1][0].y - slots[0][0].y) / scale, 0]
    : [0, 0, 0]));
  // These raw coordinates are centered over a whole recording in prepareSequence,
  // preserving trajectory while making recording location irrelevant.
  features.push(present.reduce((s, h) => s + h[0].x, 0) / present.length,
    present.reduce((s, h) => s + h[0].y, 0) / present.length, scale);
  return features;
}

export function resample(sequence: Sequence, length = FRAME_COUNT): Sequence {
  if (!sequence.length) return [];
  return Array.from({ length }, (_, i) => sequence[Math.round(i * (sequence.length - 1) / (length - 1))].slice());
}

export function prepareSequence(sequence: Sequence): Sequence {
  if (!sequence.length) return [];
  const meanScale = Math.max(0.02, sequence.reduce((sum, frame) => sum + frame[133], 0) / sequence.length);
  const startX = sequence[0][131], startY = sequence[0][132];
  return resample(sequence).map(frame => {
    frame[131] = (frame[131] - startX) / meanScale * 2;
    frame[132] = (frame[132] - startY) / meanScale * 2;
    frame[133] = (frame[133] / meanScale - 1) * 0.35;
    return frame;
  });
}

function frameDistance(a: Frame, b: Frame): number {
  let sum = 0;
  for (let i = 0; i < FEATURE_SIZE; i++) sum += (a[i] - b[i]) ** 2;
  // Missing a hand must not be mistaken for a subtly different hand shape.
  return Math.sqrt(sum / FEATURE_SIZE) + (a[0] !== b[0] || a[64] !== b[64] ? 0.4 : 0);
}

// Dynamic time warping tolerates modest changes in signing speed.
export function sequenceDistance(a: Sequence, b: Sequence): number {
  if (!a.length || !b.length) return Infinity;
  let previous = Array(b.length + 1).fill(Infinity) as number[];
  previous[0] = 0;
  for (let i = 1; i <= a.length; i++) {
    const current = Array(b.length + 1).fill(Infinity) as number[];
    for (let j = 1; j <= b.length; j++) {
      current[j] = frameDistance(a[i - 1], b[j - 1]) + Math.min(previous[j], current[j - 1], previous[j - 1]);
    }
    previous = current;
  }
  return previous[b.length] / Math.max(a.length, b.length);
}

export function recognize(sequence: Sequence, signs: Sign[]): { sign: Sign; score: number; distance: number } | null {
  if (sequence.length < 6) return null;
  const query = prepareSequence(sequence);
  const ranked = signs.filter(s => s.samples.length >= 3).map(sign => {
    const distances = sign.samples.map(sample => sequenceDistance(query, sample)).sort((a, b) => a - b);
    return { sign, distance: (distances[0] + distances[1]) / 2 };
  }).sort((a, b) => a.distance - b.distance);
  if (!ranked.length || ranked[0].distance > 0.24) return null;
  // Abstain when two different labels are almost equally similar.
  if (ranked[1] && ranked[1].distance - ranked[0].distance < 0.035) return null;
  return { ...ranked[0], score: Math.round(Math.exp(-ranked[0].distance * 4) * 100) };
}

export class CommitGate {
  private candidate = '';
  private since = 0;
  private latched = '';
  private absentSince: number | null = null;
  reset() { this.candidate = ''; this.since = 0; this.latched = ''; this.absentSince = null; }
  update(id: string | null, hasHands: boolean, now: number, duration = 900): { commit: boolean; progress: number } {
    if (!hasHands) {
      this.absentSince ??= now;
      if (now - this.absentSince >= 350) this.latched = '';
    } else this.absentSince = null;
    if (!id) { this.candidate = ''; return { commit: false, progress: 0 }; }
    if (this.candidate !== id) { this.candidate = id; this.since = now; }
    if (this.latched === id) return { commit: false, progress: 1 };
    const progress = Math.min(1, (now - this.since) / duration);
    if (progress < 1) return { commit: false, progress };
    this.latched = id;
    return { commit: true, progress: 1 };
  }
}

export function appendSign(text: string, sign: Pick<Sign, 'kind' | 'label'>): string {
  return sign.kind === 'letter' ? text + sign.label : `${text.trimEnd()}${text.trim() ? ' ' : ''}${sign.label} `;
}

export function validateLibrary(input: unknown): Library {
  if (!input || typeof input !== 'object') throw new Error('This is not a SignBridge library.');
  const data = input as Library;
  if (data.version !== 1 || !Array.isArray(data.signs) || data.signs.length > 100) throw new Error('Unsupported library format (maximum 100 signs).');
  const ids = new Set<string>(), labels = new Set<string>();
  for (const sign of data.signs) {
    if (!sign || typeof sign.id !== 'string' || !sign.id || sign.id.length > 100 || ids.has(sign.id)) throw new Error('Invalid or duplicate sign ID.');
    ids.add(sign.id);
    if ((sign.kind !== 'letter' && sign.kind !== 'word') || typeof sign.label !== 'string' || !/^[A-Z][A-Z '-]{0,39}$/.test(sign.label) || labels.has(sign.label)) throw new Error('Invalid or duplicate sign label.');
    if (sign.kind === 'letter' && sign.label.length !== 1) throw new Error('Letters must be A–Z.');
    labels.add(sign.label);
    if (!Array.isArray(sign.samples) || !sign.samples.length || sign.samples.length > MAX_SAMPLES) throw new Error('Invalid sample count.');
    for (const sequence of sign.samples) {
      if (!Array.isArray(sequence) || sequence.length !== FRAME_COUNT) throw new Error('Invalid recording length.');
      for (const frame of sequence) {
        if (!Array.isArray(frame) || frame.length !== FEATURE_SIZE || frame.some(v => typeof v !== 'number' || !Number.isFinite(v) || Math.abs(v) > 100)) throw new Error('Invalid hand landmark data.');
        if (![0, 2].includes(frame[0]) || ![0, 2].includes(frame[64]) || (!frame[0] && !frame[64])) throw new Error('A sample must contain a hand.');
      }
    }
  }
  return data;
}
