import { useCallback, useEffect, useRef, useState } from 'react';
import { FilesetResolver, HandLandmarker, PoseLandmarker } from '@mediapipe/tasks-vision';
import type { HandLandmarkerResult, PoseLandmarkerResult } from '@mediapipe/tasks-vision';

const connections = [[0,1],[1,2],[2,3],[3,4],[0,5],[5,6],[6,7],[7,8],[5,9],[9,10],[10,11],[11,12],[9,13],[13,14],[14,15],[15,16],[13,17],[0,17],[17,18],[18,19],[19,20]];

export function useCamera(onFrame: (result: HandLandmarkerResult, now: number, pose?: PoseLandmarkerResult) => void) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const stream = useRef<MediaStream | null>(null);
  const tracker = useRef<HandLandmarker | null>(null);
  const poseTracker = useRef<PoseLandmarker | null>(null);
  const trackPose = useRef(false);
  const wordGeneration = useRef(0);
  const frameId = useRef(0);
  const callback = useRef(onFrame);
  callback.current = onFrame;
  const generation = useRef(0);
  const starting = useRef(false);
  const [status, setStatus] = useState<'off' | 'loading' | 'live'>('off');
  const [error, setError] = useState('');
  const [fps, setFps] = useState(0);
  const [overlay, setOverlay] = useState(true);
  const overlayRef = useRef(overlay);
  overlayRef.current = overlay;

  const stop = useCallback(() => {
    generation.current++;
    wordGeneration.current++;
    starting.current = false;
    cancelAnimationFrame(frameId.current);
    stream.current?.getTracks().forEach(t => t.stop());
    stream.current = null;
    if (videoRef.current) videoRef.current.srcObject = null;
    tracker.current?.close();
    tracker.current = null;
    poseTracker.current?.close(); poseTracker.current = null; trackPose.current = false;
    canvasRef.current?.getContext('2d')?.clearRect(0, 0, 960, 720);
    setStatus('off'); setFps(0);
  }, []);

  const start = useCallback(async () => {
    if (starting.current || stream.current) return;
    starting.current = true;
    const session = ++generation.current;
    setError(''); setStatus('loading');
    let media: MediaStream | null = null;
    let model: HandLandmarker | null = null;
    try {
      if (!window.isSecureContext || !navigator.mediaDevices?.getUserMedia) throw new Error('Camera access needs localhost or an HTTPS connection.');
      media = await navigator.mediaDevices.getUserMedia({ video: { width: { ideal: 960 }, height: { ideal: 720 }, facingMode: 'user' }, audio: false });
      if (session !== generation.current) { media.getTracks().forEach(t => t.stop()); return; }
      stream.current = media;
      const vision = await FilesetResolver.forVisionTasks('/vision');
      const options = { runningMode: 'VIDEO' as const, numHands: 2, minHandDetectionConfidence: 0.6, minHandPresenceConfidence: 0.6, minTrackingConfidence: 0.6 };
      try { model = await HandLandmarker.createFromOptions(vision, { ...options, baseOptions: { modelAssetPath: '/vision/hand_landmarker.task', delegate: 'GPU' } }); }
      catch { model = await HandLandmarker.createFromOptions(vision, { ...options, baseOptions: { modelAssetPath: '/vision/hand_landmarker.task', delegate: 'CPU' } }); }
      if (session !== generation.current) { model.close(); media.getTracks().forEach(t => t.stop()); return; }
      tracker.current = model;
      const video = videoRef.current;
      if (!video) throw new Error('Camera preview could not be opened. Try again.');
      video.srcObject = media;
      await video.play();
      if (session !== generation.current) return;
      setStatus('live');
      let previousTime = -1, lastDetect = 0, count = 0, fpsStart = performance.now();
      const loop = (now: number) => {
        if (session !== generation.current) return;
        if (video.readyState >= 2 && video.currentTime !== previousTime && now - lastDetect >= 80) {
          previousTime = video.currentTime; lastDetect = now;
          try {
            const result = model!.detectForVideo(video, now);
            const pose = trackPose.current ? poseTracker.current?.detectForVideo(video, now) : undefined;
            const canvas = canvasRef.current;
            if (canvas) {
              canvas.width = video.videoWidth; canvas.height = video.videoHeight;
              const ctx = canvas.getContext('2d');
              if (ctx && overlayRef.current) {
                result.landmarks.forEach((hand, index) => {
                  ctx.strokeStyle = index === 0 ? '#c6f395' : '#87d9ed'; ctx.lineWidth = 3;
                  for (const [a, b] of connections) {
                    ctx.beginPath(); ctx.moveTo(hand[a].x * canvas.width, hand[a].y * canvas.height);
                    ctx.lineTo(hand[b].x * canvas.width, hand[b].y * canvas.height); ctx.stroke();
                  }
                  ctx.fillStyle = '#fff';
                  hand.forEach(point => { ctx.beginPath(); ctx.arc(point.x * canvas.width, point.y * canvas.height, 4, 0, Math.PI * 2); ctx.fill(); });
                });
              }
            }
            callback.current(result, now, pose); count++;
            if (now - fpsStart >= 1000) { setFps(Math.round(count * 1000 / (now - fpsStart))); count = 0; fpsStart = now; }
          } catch {
            stop(); setError('Hand tracking stopped unexpectedly. Restart the camera to try again.'); return;
          }
        }
        frameId.current = requestAnimationFrame(loop);
      };
      frameId.current = requestAnimationFrame(loop);
      media.getVideoTracks()[0].onended = () => {
        if (session === generation.current) { stop(); setError('The camera disconnected. Connect it and try again.'); }
      };
    } catch (e) {
      media?.getTracks().forEach(t => t.stop());
      if (session !== generation.current) { model?.close(); return; }
      stop();
      const name = e instanceof DOMException ? e.name : '';
      setError(name === 'NotAllowedError' ? 'Camera permission was denied. Allow camera access in your browser, then try again.'
        : name === 'NotFoundError' ? 'No camera found. Connect a webcam and try again.'
        : name === 'NotReadableError' ? 'The camera is busy. Close other apps using it and try again.'
        : e instanceof Error && /Camera access|preview/.test(e.message) ? e.message
        : 'The hand tracker could not load. Run npm run setup:vision, then restart the camera.');
    } finally { if (session === generation.current) starting.current = false; }
  }, [stop]);

  const startWordTracking = useCallback(async () => {
    const session = generation.current;
    const wordSession = ++wordGeneration.current;
    if (!stream.current) return false;
    if (!poseTracker.current) {
      const vision = await FilesetResolver.forVisionTasks('/vision');
      const options = { runningMode: 'VIDEO' as const, numPoses: 1, minPoseDetectionConfidence: .6, minPosePresenceConfidence: .6, minTrackingConfidence: .6, outputSegmentationMasks: false };
      let model: PoseLandmarker;
      try { model = await PoseLandmarker.createFromOptions(vision, { ...options, baseOptions: { modelAssetPath: '/vision/pose_landmarker.task', delegate: 'GPU' } }); }
      catch { model = await PoseLandmarker.createFromOptions(vision, { ...options, baseOptions: { modelAssetPath: '/vision/pose_landmarker.task', delegate: 'CPU' } }); }
      if (session !== generation.current || wordSession !== wordGeneration.current) { model.close(); return false; }
      poseTracker.current = model;
    }
    trackPose.current = true;
    return true;
  }, []);
  const stopWordTracking = useCallback(() => { wordGeneration.current++; trackPose.current = false; }, []);

  useEffect(() => () => stop(), [stop]);
  useEffect(() => {
    const hide = () => { if (document.hidden) stop(); };
    document.addEventListener('visibilitychange', hide);
    return () => document.removeEventListener('visibilitychange', hide);
  }, [stop]);

  return { videoRef, canvasRef, status, error, fps, start, stop, overlay, setOverlay, startWordTracking, stopWordTracking };
}
