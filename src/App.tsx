import { useEffect, useRef, useState } from 'react';
import type { HandLandmarkerResult, PoseLandmarkerResult } from '@mediapipe/tasks-vision';
import { ArrowDownToLine, ArrowRight, AudioLines, BookOpen, Check, ChevronDown, ChevronRight, CircleHelp, Copy, Download, Hand, History, Keyboard, Layers3, Lightbulb, LoaderCircle, Monitor, Plus, ScanLine, Settings2, ShieldCheck, Sparkles, Square, Trash2, Upload, Video, Volume2, X } from 'lucide-react';
import { appendSign, CommitGate, extractFeatures, MAX_SAMPLES, prepareSequence, recognize, STORAGE_KEY, validateLibrary } from './recognition';
import type { Library, Sequence, Sign } from './recognition';
import { useCamera } from './useCamera';
import { loadLetterRecognizer } from './builtinLetters';
import type { LetterRecognizer } from './builtinLetters';
import type { WordRecognizer } from './builtinWords';
import { wordFrame } from './wordFeatures';
import type { WordFrame } from './wordFeatures';
import { SignReference } from './SignReference';

type Page = 'translate' | 'teach' | 'library' | 'guide';
const alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'.split('');
const emptyLibrary: Library = { version: 1, signs: [] };
function loadLibrary() {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    return { library: raw ? validateLibrary(JSON.parse(raw)) : emptyLibrary, error: '' };
  } catch { return { library: emptyLibrary, error: 'Your saved library could not be read. Import a backup or record your signs again.' }; }
}
function download(data: string, name: string, type = 'text/plain') {
  const url = URL.createObjectURL(new Blob([data], { type }));
  const a = document.createElement('a'); a.href = url; a.download = name; a.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

function DemoHands() {
  return <svg className="demo-hands" viewBox="0 0 600 380" aria-hidden="true">
    <defs><linearGradient id="handShade" x1="0" y1="0" x2="1" y2="1"><stop stopColor="#ffd5a8"/><stop offset="1" stopColor="#ffb8c4"/></linearGradient></defs>
    <circle cx="300" cy="193" r="142" fill="none" stroke="#fff" strokeOpacity=".65"/><circle cx="300" cy="193" r="110" fill="none" stroke="#fff" strokeOpacity=".38"/>
    <path d="M222 310c-8-34-13-49-27-69l-32-47c-12-17-4-31 9-25l29 35-22-105c-4-21 18-24 22-5l21 80-6-114c-1-19 22-21 23-1l7 105 12-95c3-22 24-19 21 4l-10 93 18-57c7-20 27-12 21 6l-22 91c-7 38-5 60-2 104" fill="url(#handShade)" stroke="#ef9a78" strokeWidth="2"/>
    <path d="M365 314c4-36 12-54 31-77l35-44c12-16 3-30-10-22l-30 33 23-105c4-21-18-24-22-5l-22 80 7-114c1-19-22-21-24-1l-6 105-12-95c-3-22-24-19-21 4l10 93-18-57c-7-20-27-12-21 6l22 91c7 38 5 60 2 104" fill="url(#handShade)" stroke="#ef9a78" strokeWidth="2"/>
    <g fill="none" stroke="#89c9d7" strokeWidth="2.5" strokeOpacity=".9"><path d="M244 280 220 201 181 180M244 280 228 166 191 100M244 280 245 161 226 62M244 280 265 170 272 82M244 280 282 205 300 145M354 280 376 201 423 183M354 280 370 166 402 100M354 280 353 161 376 62M354 280 333 170 330 82M354 280 316 205 302 145"/></g>
    <g fill="#ffe082">{[[244,280],[220,201],[181,180],[228,166],[191,100],[245,161],[226,62],[265,170],[272,82],[282,205],[300,145],[354,280],[376,201],[423,183],[370,166],[402,100],[353,161],[376,62],[333,170],[330,82],[316,205]].map(([cx,cy],i)=><circle key={i} cx={cx} cy={cy} r="4.5"/>)}</g>
  </svg>;
}

export default function App() {
  const [loaded] = useState(loadLibrary);
  const [library, setLibrary] = useState<Library>(loaded.library);
  const [page, setPage] = useState<Page>('translate');
  const [notice, setNotice] = useState(loaded.error);
  const [transcript, setTranscript] = useState('');
  const [detected, setDetected] = useState<{ label: string; score: number; kind: 'letter' | 'word'; scoreType?:'match' } | null>(null);
  const [recognitionMode, setRecognitionMode] = useState<'letters' | 'words' | 'personal'>('letters');
  const [letterReady, setLetterReady] = useState(false);
  const [wordReady, setWordReady] = useState(false);
  const [modelError, setModelError] = useState('');
  const letterModel = useRef<LetterRecognizer | null>(null);
  const wordModel = useRef<WordRecognizer | null>(null);
  const modeRef = useRef(recognitionMode); modeRef.current = recognitionMode;
  const [wordCapture, setWordCapture] = useState<{start:number;end:number} | null>(null);
  const [wordBusy, setWordBusy] = useState(false);
  const [wordPhase, setWordPhase] = useState('');
  const wordCaptureRef = useRef(wordCapture); wordCaptureRef.current = wordCapture;
  const wordFrames = useRef<WordFrame[]>([]);
  const wordFrameCount = useRef(0);
  const wordHandFrameCount = useRef(0);
  const wordRequest = useRef(0);
  const [hands, setHands] = useState(0);
  const [progress, setProgress] = useState(0);
  const [demo, setDemo] = useState(false);
  const [demoRun, setDemoRun] = useState(0);
  const [autoSpeak, setAutoSpeak] = useState(false);
  const [speakLetters, setSpeakLetters] = useState(false);
  const [paused, setPaused] = useState(false);
  const [speaking, setSpeaking] = useState(false);
  const [savedPhrases, setSavedPhrases] = useState<string[]>([]);
  const [trainingKind, setTrainingKind] = useState<'letter' | 'word'>('letter');
  const [trainingLabel, setTrainingLabel] = useState('A');
  const [wordLabel, setWordLabel] = useState('');
  const [recording, setRecording] = useState<{ sign: Pick<Sign, 'label' | 'kind'>; start: number; end: number } | null>(null);
  const [recordPhase, setRecordPhase] = useState('');
  const [recordProgress, setRecordProgress] = useState(0);
  const [deleteId, setDeleteId] = useState<string | null>(null);
  const [showSettings, setShowSettings] = useState(false);
  const libraryRef = useRef(library); libraryRef.current = library;
  const pageRef = useRef(page); pageRef.current = page;
  const optionsRef = useRef({ autoSpeak, speakLetters, paused }); optionsRef.current = { autoSpeak, speakLetters, paused };
  const recordingRef = useRef(recording); recordingRef.current = recording;
  const recordFrames = useRef<Sequence>([]);
  const sampleFrames = useRef<{ time: number; frame: number[] }[]>([]);
  const gate = useRef(new CommitGate());
  const lastPrediction = useRef(0);
  const lastSeen = useRef(0);
  const importRef = useRef<HTMLInputElement>(null);
  const modalRef = useRef<HTMLDivElement>(null);
  const readySigns = library.signs.filter(s => s.samples.length >= 3);
  const activeLabel = trainingKind === 'letter' ? trainingLabel : wordLabel.trim().toUpperCase();
  const selectedSign = library.signs.find(s => s.label === activeLabel);

  function persist(next: Library) {
    setLibrary(next);
    libraryRef.current = next;
    try { localStorage.setItem(STORAGE_KEY, JSON.stringify(next)); }
    catch { setNotice('Browser storage is unavailable or full. Export your sign library to keep this session’s recordings.'); }
  }
  function speak(text: string) {
    if (!text.trim()) return;
    if (!('speechSynthesis' in window)) { setNotice('Speech is unavailable in this browser. You can copy or download the text.'); return; }
    speechSynthesis.cancel();
    const utterance = new SpeechSynthesisUtterance(text.trim());
    utterance.lang = 'en-IN'; utterance.rate = 0.85;
    const voice = speechSynthesis.getVoices().find(v => v.lang === 'en-IN') || speechSynthesis.getVoices().find(v => v.lang.startsWith('en'));
    if (voice) utterance.voice = voice;
    utterance.onstart = () => setSpeaking(true);
    utterance.onend = () => setSpeaking(false);
    utterance.onerror = (e) => { setSpeaking(false); if (e.error !== 'canceled' && e.error !== 'interrupted') setNotice('Speech could not play. Check your browser audio settings and try again.'); };
    speechSynthesis.speak(utterance);
  }
  function commit(sign: Pick<Sign, 'label' | 'kind'>) {
    setTranscript(text => appendSign(text, sign).slice(0, 5000));
    if ((sign.kind === 'word' && optionsRef.current.autoSpeak) || (sign.kind === 'letter' && optionsRef.current.speakLetters)) speak(sign.label);
  }
  function onFrame(result: HandLandmarkerResult, now: number, pose?: PoseLandmarkerResult) {
    setHands(result.landmarks.length);
    if(wordCaptureRef.current && now >= wordCaptureRef.current.start && now < wordCaptureRef.current.end) {
      wordFrameCount.current++;
      if(result.landmarks.length) wordHandFrameCount.current++;
      const feature = wordFrame(pose?.landmarks[0] ?? [],result.landmarks);
      if(feature) wordFrames.current.push(feature);
    }
    const frame = extractFeatures(result.landmarks, result.handedness.map(h => h[0]?.categoryName ?? 'Right'));
    if (recordingRef.current && now >= recordingRef.current.start && now < recordingRef.current.end && frame) recordFrames.current.push(frame);
    if(pageRef.current === 'translate' && modeRef.current === 'words') return;
    if (!frame) {
      if (now - lastSeen.current > 300) sampleFrames.current = [];
      gate.current.update(null, false, now); setDetected(null); setProgress(0); return;
    }
    lastSeen.current = now;
    sampleFrames.current.push({ time: now, frame });
    sampleFrames.current = sampleFrames.current.filter(f => now - f.time <= 1800);
    if (pageRef.current !== 'translate' || optionsRef.current.paused) { gate.current.reset(); setDetected(null); setProgress(0); return; }
    if (now - lastPrediction.current < 220) return;
    lastPrediction.current = now;
    const personal = modeRef.current === 'personal' ? recognize(sampleFrames.current.map(f => f.frame), libraryRef.current.signs) : null;
    const builtin = modeRef.current === 'letters' ? letterModel.current?.predict(result.landmarks,camera.videoRef.current?.videoWidth ?? 960,camera.videoRef.current?.videoHeight ?? 720) : null;
    const prediction = personal ? { ...personal.sign, score: personal.score, scoreType:'match' as const } : builtin;
    setDetected(prediction ?? null);
    const resultGate = gate.current.update(prediction?.id ?? null, true, now);
    setProgress(resultGate.progress);
    if (resultGate.commit && prediction) commit(prediction);
  }
  const camera = useCamera(onFrame);
  const live = camera.status === 'live';
  const loading = camera.status === 'loading';

  function resetRecognition() { gate.current.reset(); sampleFrames.current = []; setDetected(null); setProgress(0); setHands(0); }
  function cancelWordCapture() { wordRequest.current++; wordCaptureRef.current=null; setWordCapture(null); setWordBusy(false); setWordPhase(''); camera.stopWordTracking(); }
  function stopAll() { setDemo(false); setPaused(false); cancelWordCapture(); camera.stop(); setRecording(null); resetRecognition(); }
  function startCamera() { if(demo)setTranscript(''); setDemo(false); setPaused(false); resetRecognition(); void camera.start(); }
  function navigate(next: Page) {
    if (recording) setRecording(null);
    cancelWordCapture();
    setDemo(false); setPaused(false); resetRecognition();
    if (next === 'library' || next === 'guide') camera.stop();
    setPage(next);
  }
  function startDemo() { cancelWordCapture(); camera.stop(); resetRecognition(); setTranscript(''); setPage('translate'); setRecognitionMode('letters'); setDemo(true); setDemoRun(n => n + 1); }

  function changeRecognitionMode(mode: typeof recognitionMode) { cancelWordCapture(); resetRecognition(); setDemo(false); setModelError(''); setRecognitionMode(mode); setPaused(false); }
  async function startWordCapture() {
    if(!live || paused || wordBusy || wordCapture) return;
    const request = ++wordRequest.current;
    setWordBusy(true); setWordPhase('Loading word tracker…'); setNotice(''); setDetected(null); setProgress(0);
    try {
      const { loadWordRecognizer } = await import('./builtinWords');
      const model = await loadWordRecognizer();
      if(request !== wordRequest.current) return;
      wordModel.current = model; setWordReady(true);
      if(!await camera.startWordTracking()) throw new Error('Start the camera again before capturing a word.');
      if(request !== wordRequest.current) { camera.stopWordTracking(); return; }
      wordFrames.current=[]; wordFrameCount.current=0; wordHandFrameCount.current=0;
      const start=performance.now()+1500;
      setWordCapture({start,end:start+4000});
    } catch(e) { if(request === wordRequest.current) setNotice(e instanceof Error?e.message:'Word tracking could not start. Try again.'); }
    finally { if(request === wordRequest.current) setWordBusy(false); }
  }

  useEffect(() => {
    let active=true;
    void loadLetterRecognizer().then(model=>{if(active){letterModel.current=model;setLetterReady(true);}}).catch(()=>{if(active)setModelError('The alphabet model could not load. Refresh the page and try again.');});
    return () => {active=false;};
  }, []);
  useEffect(() => {
    if(recognitionMode !== 'words' || wordModel.current) return;
    let active=true; setModelError('');
    void import('./builtinWords').then(module=>module.loadWordRecognizer()).then(model=>{if(active){wordModel.current=model;setWordReady(true);}}).catch(()=>{if(active)setModelError('The word model could not load. Refresh the page and try again.');});
    return () => {active=false;};
  }, [recognitionMode]);

  useEffect(() => {
    if(!wordCapture) return;
    const request=wordRequest.current;
    let finished=false;
    const tick=()=>{
      if(finished) return;
      const now=performance.now();
      if(now<wordCapture.start) {setWordPhase(`Get ready · ${Math.ceil((wordCapture.start-now)/1000)}`);return;}
      setWordPhase('Perform one word sign…');
      setProgress(Math.min(1,(now-wordCapture.start)/(wordCapture.end-wordCapture.start)));
      if(now<wordCapture.end) return;
      finished=true;
      const frames=wordFrames.current.slice();
      setWordCapture(null); camera.stopWordTracking(); setWordBusy(true); setWordPhase('Recognizing your sign…');
      void (async()=>{
        try {
          if(frames.length<12 || frames.length/Math.max(1,wordFrameCount.current)<.7 || wordHandFrameCount.current/Math.max(1,wordFrameCount.current)<.5) throw new Error('Keep your face, shoulders and signing hands visible for the full capture, then try again.');
          const prediction=await wordModel.current!.predict(frames);
          if(request!==wordRequest.current) return;
          setDetected(prediction); setProgress(prediction?1:0);
          if(prediction) commit(prediction);
          else setNotice('No clear word match. Repeat one supported ISL sign with your upper body in view.');
        }catch(e){if(request===wordRequest.current)setNotice(e instanceof Error?e.message:'Word recognition failed. Please try again.');}
        finally{if(request===wordRequest.current){setWordBusy(false);setWordPhase('');}}
      })();
    };
    tick();const timer=setInterval(tick,100);
    return()=>clearInterval(timer);
  }, [wordCapture]);

  useEffect(() => {
    if (!demo) return;
    const letters = 'HELLO WORLD'.split('');
    let index = 0;
    setDetected({ label: letters[0], kind: 'letter', score: 96 });
    const timer = setInterval(() => {
      const label = letters[index++];
      if (label === ' ') { setTranscript(t => `${t.trimEnd()} `); setDetected(null); }
      else if (label) { setDetected({ label, kind: 'letter', score: 96 }); setTranscript(t => t + label); }
      if (index >= letters.length) { clearInterval(timer); setProgress(1); }
    }, 800);
    return () => clearInterval(timer);
  }, [demo, demoRun]);

  useEffect(() => {
    if (!recording) { setRecordPhase(''); setRecordProgress(0); return; }
    const tick = () => {
      const now = performance.now();
      if (now < recording.start) { setRecordPhase(`Get ready · ${Math.ceil((recording.start - now) / 1000)}`); return; }
      setRecordPhase('Recording your sign…');
      setRecordProgress(Math.min(1, (now - recording.start) / (recording.end - recording.start)));
      if (now < recording.end) return;
      setRecording(null);
      if (recordFrames.current.length < 15) { setNotice('Recording missed too many frames. Keep your signing hand(s) visible for the full recording and try again.'); return; }
      const current = libraryRef.current;
      const existing = current.signs.find(s => s.label === recording.sign.label);
      if (existing && existing.samples.length >= MAX_SAMPLES) { setNotice('This sign already has 8 recordings. Delete it to start again.'); return; }
      const sample = prepareSequence(recordFrames.current);
      const signs = existing
        ? current.signs.map(s => s.id === existing.id ? { ...s, samples: [...s.samples, sample] } : s)
        : [...current.signs, { ...recording.sign, id: crypto.randomUUID(), samples: [sample] }];
      persist({ version: 1, signs });
      setNotice(`Saved ${recording.sign.label} · recording ${(existing?.samples.length ?? 0) + 1}. ${existing && existing.samples.length >= 2 ? 'Ready to recognize.' : 'Record at least 3 examples to enable this sign.'}`);
    };
    tick(); const timer = setInterval(tick, 100);
    return () => clearInterval(timer);
  }, [recording]);

  useEffect(() => { if (camera.status === 'off') { setRecording(null); cancelWordCapture(); resetRecognition(); } }, [camera.status]);
  useEffect(() => () => { if ('speechSynthesis' in window) speechSynthesis.cancel(); }, []);

  useEffect(() => {
    if (!deleteId && !showSettings) return;
    const previous = document.activeElement as HTMLElement | null;
    const dialog = modalRef.current;
    dialog?.querySelector<HTMLElement>('button')?.focus();
    const keydown = (event: KeyboardEvent) => {
      if (event.key === 'Escape') { setDeleteId(null); setShowSettings(false); }
      if (event.key !== 'Tab') return;
      const controls = Array.from(dialog?.querySelectorAll<HTMLElement>('button:not(:disabled), input, a[href]') ?? []);
      const first = controls[0], last = controls.at(-1);
      if (event.shiftKey && document.activeElement === first) { event.preventDefault(); last?.focus(); }
      else if (!event.shiftKey && document.activeElement === last) { event.preventDefault(); first?.focus(); }
    };
    document.addEventListener('keydown', keydown);
    return () => { document.removeEventListener('keydown', keydown); previous?.focus(); };
  }, [deleteId, showSettings]);

  function startRecording() {
    const label = activeLabel;
    if (!live) { setNotice('Start your camera before recording a sign.'); return; }
    if (!/^[A-Z][A-Z '-]{0,39}$/.test(label) || (trainingKind === 'word' && label.length < 2)) { setNotice('Enter a word label using 2–40 English letters, spaces, apostrophes or hyphens.'); return; }
    if (selectedSign && selectedSign.kind !== trainingKind) { setNotice('This label is already used. Choose a different label.'); return; }
    if (!selectedSign && library.signs.length >= 100) { setNotice('Library limit reached (100 signs). Export a backup and delete an unused sign first.'); return; }
    recordFrames.current = [];
    const start = performance.now() + 2500;
    setRecording({ sign: { label, kind: trainingKind }, start, end: start + 1800 });
  }
  async function importLibrary(file?: File) {
    if (!file) return;
    try {
      if (file.size > 32_000_000) throw new Error('Library file is too large (maximum 32 MB).');
      const data = validateLibrary(JSON.parse(await file.text()));
      // Merge without silently overwriting the user's existing recordings.
      const conflicts = data.signs.filter(s => libraryRef.current.signs.some(x => x.label === s.label));
      const additions = data.signs.filter(s => !libraryRef.current.signs.some(x => x.label === s.label)).map(s => ({ ...s, id: crypto.randomUUID() }));
      if (libraryRef.current.signs.length + additions.length > 100) throw new Error('The merged library would exceed 100 signs.');
      persist({ version: 1, signs: [...libraryRef.current.signs, ...additions] });
      setNotice(`Imported ${additions.length} signs.${conflicts.length ? ` Kept ${conflicts.length} existing labels unchanged.` : ''}`);
    } catch (e) { setNotice(e instanceof Error ? e.message : 'Could not import this library.'); }
  }
  function addSpace() {
    if (autoSpeak) { const word = transcript.trim().split(/\s+/).at(-1); if (word) speak(word); }
    setTranscript(t => t && !t.endsWith(' ') ? `${t} ` : t);
  }

  return <div className="app-shell">
    <aside className="sidebar">
      <a className="brand" href="#" onClick={e => { e.preventDefault(); navigate('translate'); }}><span className="brand-icon"><Hand size={25}/></span><span>sign<span className="brand-light">bridge</span><small>SIGNS INTO CONVERSATION</small></span></a>
      <div className="workspace-label">YOUR WORKSPACE</div>
      <nav aria-label="Main navigation">
        {([{ id: 'translate', label: 'Live translate', icon: ScanLine }, { id: 'teach', label: 'Teach signs', icon: Sparkles }, { id: 'library', label: 'My sign library', icon: Layers3 }, { id: 'guide', label: 'Getting started', icon: BookOpen }] as const).map(item => <button key={item.id} className={`nav-item ${page === item.id ? 'active' : ''}`} onClick={() => navigate(item.id)}><item.icon size={19}/>{item.label}{item.id === 'library' && <span className="nav-count">{library.signs.length}</span>}{page === item.id && <ChevronRight className="nav-arrow" size={15}/>}</button>)}
      </nav>
      <div className="sidebar-tip"><div className="tip-icon"><Lightbulb size={19}/></div><h3>Your hands.<br/>Your words.</h3><p>Use built-in ISL recognition. Add personal signs whenever you need.</p><button onClick={() => navigate('teach')}>Build your sign library <ArrowRight size={15}/></button></div>
      <div className="sidebar-bottom"><div className="privacy-mark"><ShieldCheck size={17}/><span>Hand tracking runs locally</span></div><div className="profile"><span className="profile-avatar">Y</span><div>Your personal space<small>No account needed</small></div><Settings2 size={17} onClick={() => setShowSettings(true)} role="button" tabIndex={0} aria-label="Open preferences" onKeyDown={e => { if (e.key === 'Enter') setShowSettings(true); }}/></div></div>
    </aside>
    <div className="main-shell">
      <header className="topbar"><div className="breadcrumb">Workspace <ChevronRight size={13}/><span>{{translate:'Live translate',teach:'Teach signs',library:'My sign library',guide:'Getting started'}[page]}</span></div><div className="topbar-right"><span className="language-pill"><span className="india-flag">🇮🇳</span> Indian Sign Language <ChevronDown size={13}/></span><button className="icon-button" title="Getting started" aria-label="Getting started" onClick={() => navigate('guide')}><CircleHelp size={20}/></button></div></header>
      <main>
        <div className="page-heading"><div><div className="eyebrow"><span/> A NEW WAY TO BE HEARD</div><h1>{{translate:'Let your hands do the talking.',teach:'Make it your own.',library:'Every sign, a new possibility.',guide:'Your first conversation starts here.'}[page]}</h1><p>{{translate:'Recognize ISL letters and word signs, then turn your message into a voice.',teach:'Record your ISL signs to create a library that recognizes the way you sign.',library:'Your personal collection of letters and words, saved in this browser.',guide:'A few simple steps to bring your signs to the screen.'}[page]}</p></div><span className="prototype-tag"><span/> ISL RECOGNITION PROTOTYPE</span></div>
        {notice && <div className="notice" role="status"><CircleHelp size={17}/><span>{notice}</span><button onClick={() => setNotice('')} aria-label="Dismiss notification"><X size={17}/></button></div>}

        {(page === 'translate' || page === 'teach') && <>
          <div className="workspace-grid">
            <section className="camera-card">
              <div className="section-header"><h2><Video size={18}/> {page === 'teach' ? 'Recording studio' : 'Camera workspace'}</h2><span className={`status-badge ${live ? 'is-live' : ''}`}><span/>{demo ? 'SIMULATED DEMO' : live ? 'CAMERA LIVE' : loading ? 'CONNECTING' : 'CAMERA OFF'}</span></div>
              <div className={`camera-stage ${live ? 'with-camera' : ''}`}>
                <video ref={camera.videoRef} autoPlay playsInline muted className={live ? '' : 'hidden-video'}/><canvas ref={camera.canvasRef} className={live ? '' : 'hidden-video'}/>
                <div className="view-label"><span/>{demo ? 'DEMO PREVIEW' : 'LIVE PREVIEW'}</div><div className="frame-corner top-left"/><div className="frame-corner top-right"/><div className="frame-corner bottom-left"/><div className="frame-corner bottom-right"/>
                {!live && <div className="camera-empty"><DemoHands/><div className="camera-empty-content"><span className="camera-glyph">{loading ? <LoaderCircle size={21} className="spin"/> : <Video size={21}/>}</span><h3>{demo ? 'A glimpse of what’s possible.' : loading ? 'Getting your camera ready…' : 'Your next conversation starts here.'}</h3><p>{demo ? 'Simulated letters show how text builds as you sign.' : loading ? 'Loading the local hand tracker. This can take a few seconds.' : 'Enable your camera, keep your hands in view, and start signing.'}</p>{!demo && !loading && <button className="primary-button" onClick={startCamera}><Video size={17}/> Enable camera <ArrowRight size={16}/></button>}{loading && <button className="text-on-dark" onClick={stopAll}>Cancel</button>}{demo && <button className="primary-button" onClick={startCamera}><Video size={17}/> Try with your camera</button>}</div></div>}
                {live && <div className="camera-live-hint">{recording ? recordPhase : wordCapture ? wordPhase : page === 'teach' ? 'Keep both hands comfortably inside the frame' : paused ? 'Recognition paused' : recognitionMode === 'words' ? (wordCapture ? wordPhase : 'Show your upper body · use Recognize a word') : recognitionMode === 'personal' && !readySigns.length ? 'Teach a sign to enable personal recognition' : hands ? `${hands} hand${hands > 1 ? 's' : ''} in view · ${detected ? 'hold your sign' : 'show a trained sign'}` : 'Bring your signing hands into view'}</div>}
                {(recording || wordCapture) && <div className="recording-bar"><span style={{width:`${(recording ? recordProgress : progress) * 100}%`}}/></div>}
                <div className="camera-meta"><span><ShieldCheck size={13}/> Video stays on your device</span><span>{live ? `${camera.fps} FPS` : '2-hand tracking'} <span className="meta-dot">·</span> {demo ? 'Illustration' : 'MediaPipe'}</span></div>
              </div>
              <div className="camera-toolbar"><button className="toolbar-button" disabled={!live || !!wordCapture || wordBusy} onClick={() => { setPaused(!paused); gate.current.reset(); }}><span className="pause-symbol">{paused ? '▶' : 'Ⅱ'}</span>{paused ? 'Resume' : 'Pause'} recognition</button><button className={`toolbar-button ${camera.overlay ? 'selected' : ''}`} onClick={() => camera.setOverlay(!camera.overlay)} aria-pressed={camera.overlay}><ScanLine size={16}/> Hand landmarks</button><button className="toolbar-button stop-button" onClick={stopAll} disabled={!live && !loading && !demo}><Square size={13}/> Stop</button></div>
              {camera.error && <div className="camera-error" role="alert">{camera.error}</div>}
              <div className="camera-footnote"><Lightbulb size={16}/><span>Good light. A clear background. Both hands in frame.</span></div>
              <SignReference
                key={page === 'teach' ? `teach-${trainingKind}` : `translate-${recognitionMode === 'words' || recognitionMode === 'personal' && detected?.kind === 'word' ? 'word' : 'letter'}`}
                mode={page === 'teach' ? trainingKind : recognitionMode === 'words' || recognitionMode === 'personal' && detected?.kind === 'word' ? 'word' : 'letter'}
                initialLetter={page === 'teach' && trainingKind === 'letter' ? trainingLabel : detected?.kind === 'letter' ? detected.label : undefined}
                wordLabel={page === 'teach' && trainingKind === 'word' ? activeLabel : detected?.kind === 'word' ? detected.label : undefined}
              />
            </section>

            {page === 'translate' ? <section className="translation-card">
              <div className="section-header"><h2><AudioLines size={18}/> Your translation</h2><span className="live-text-tag">{demo ? 'DEMO' : 'LIVE TEXT'}</span></div>
              <div className="recognition-controls">
                <div className="segmented recognition-mode" aria-label="Recognition mode">
                  <button className={recognitionMode==='letters'?'chosen':''} onClick={()=>changeRecognitionMode('letters')} aria-pressed={recognitionMode==='letters'}>Letters</button>
                  <button className={recognitionMode==='words'?'chosen':''} onClick={()=>changeRecognitionMode('words')} aria-pressed={recognitionMode==='words'}>Words</button>
                  <button className={recognitionMode==='personal'?'chosen':''} onClick={()=>changeRecognitionMode('personal')} aria-pressed={recognitionMode==='personal'}>My signs</button>
                </div>
                <p className="model-status">{modelError || (recognitionMode==='letters' ? letterReady?'Built-in ISL A–Z · no training needed':'Loading alphabet model…' : recognitionMode==='words' ? wordReady?'Built-in ISL words · no training needed':'Loading word model…' : `${readySigns.length} personal signs ready`)}</p>
                {recognitionMode==='words' && <div className="word-capture-controls"><p>Keep your face, shoulders and hands in view. Perform one ISL word sign during the 4-second capture.</p><button className="primary-button" onClick={()=>void startWordCapture()} disabled={!live || !wordReady || !!wordCapture || wordBusy || paused}>{wordBusy?<LoaderCircle size={16} className="spin"/>:<ScanLine size={16}/>} {wordCapture ? wordPhase : wordBusy ? wordPhase : 'Recognize a word'}</button>{(wordCapture||wordBusy)&&<button className="cancel-record" onClick={cancelWordCapture}>Cancel</button>}{wordReady && <details className="supported-words"><summary>Supported word signs ({new Set(wordModel.current?.metadata.classes).size})</summary><div>{Array.from(new Set(wordModel.current?.metadata.classes)).sort().map(label=><span key={label}>{label}</span>)}</div></details>}</div>}
              </div>
              <div className="detected-block"><div className="mini-label">{recognitionMode === 'words' && !demo || detected?.kind === 'word' ? 'DETECTED WORD' : 'DETECTED LETTER'}<span className={detected ? 'match-active' : ''}>{detected ? `${detected.score}% ${detected.scoreType === 'match' || recognitionMode === 'personal' || demo ? 'match' : 'model score'}` : 'Waiting for a sign'}</span></div><div className={`detected-sign ${detected?.kind === 'word' ? 'word-sign' : ''}`}>{detected?.label ?? <span className="waiting-dash">—</span>}</div><div className="hold-progress"><span style={{width:`${progress * 100}%`}}/></div><p>{demo ? 'Sample output · no camera recognition' : wordCapture ? wordPhase : progress === 1 ? recognitionMode === 'words' ? 'Word added. Capture another sign to continue.' : 'Added. Lower your hands to repeat this sign.' : detected ? 'Hold your letter steadily to add it to your text.' : recognitionMode === 'words' ? 'Capture one supported word sign at a time.' : recognitionMode === 'letters' ? 'Built-in A–Z recognition. Hold a clear ISL letter sign.' : !readySigns.length ? 'Record 3 examples of a sign to get started.' : 'Recognizes signs from your personal library.'}</p></div>
              <div className="text-block"><div className="mini-label">YOUR MESSAGE <span>{transcript.trim() ? transcript.trim().split(/\s+/).length : 0} words</span></div><textarea aria-label="Your message" value={transcript} onChange={e => setTranscript(e.target.value)} placeholder="Your words will appear here…" maxLength={5000}/><div className="editing-tools"><button onClick={addSpace} title="Finish the current word"><Keyboard size={15}/> Space</button><button onClick={() => { setTranscript(t => t.slice(0,-1)); gate.current.reset(); }} disabled={!transcript} title="Remove the last character">⌫ <span>Delete</span></button><button onClick={() => { setTranscript(''); gate.current.reset(); }} disabled={!transcript}><Trash2 size={14}/> Clear</button></div></div>
              <div className="speech-tools"><button className="speak-button" disabled={!transcript.trim()} onClick={() => { if (speaking) { speechSynthesis.cancel(); setSpeaking(false); } else speak(transcript); }}><Volume2 size={18}/>{speaking ? 'Stop speaking' : 'Speak message'}<AudioLines size={18}/></button><label className="toggle-row"><span><span className="toggle-title">Speak words automatically</span><small>When a word sign is added or you press Space</small></span><input type="checkbox" checked={autoSpeak} onChange={e => setAutoSpeak(e.target.checked)}/><span className="switch"/></label></div>
              <div className="output-tools"><button disabled={!transcript.trim()} onClick={() => { void navigator.clipboard?.writeText(transcript).then(() => setNotice('Message copied.')).catch(() => setNotice('Could not copy. Select the message text or download it.')); if (!navigator.clipboard) setNotice('Clipboard is unavailable. Use Download instead.'); }}><Copy size={15}/> Copy</button><button disabled={!transcript.trim()} onClick={() => download(transcript, 'signbridge-message.txt')}><Download size={15}/> Download</button><button disabled={!transcript.trim()} onClick={() => { setSavedPhrases(p => [transcript.trim(),...p.filter(x => x !== transcript.trim())].slice(0,20)); setNotice('Message saved for this session.'); }}><Plus size={15}/> Save</button></div>
            </section> : <section className="training-card"><div className="section-header"><h2><Sparkles size={18}/> Teach a new sign</h2><span className="live-text-tag">PERSONAL</span></div><div className="training-content"><p className="training-intro">You show the sign.<br/><strong>SignBridge learns the pattern.</strong></p><div className="segmented"><button className={trainingKind === 'letter' ? 'chosen' : ''} onClick={() => setTrainingKind('letter')} disabled={!!recording}>Letter</button><button className={trainingKind === 'word' ? 'chosen' : ''} onClick={() => setTrainingKind('word')} disabled={!!recording}>Word / short phrase</button></div>{trainingKind === 'letter' ? <><label className="field-label" htmlFor="letter-select">Which letter are you signing?</label><select id="letter-select" value={trainingLabel} onChange={e => setTrainingLabel(e.target.value)} disabled={!!recording}>{alphabet.map(a => <option key={a} value={a}>{a}{library.signs.find(s=>s.label === a) ? ' · recorded' : ''}</option>)}</select></> : <><label className="field-label" htmlFor="word-label">Name your sign</label><input className="word-input" id="word-label" placeholder="e.g. HELLO or THANK YOU" value={wordLabel} onChange={e => setWordLabel(e.target.value)} disabled={!!recording} maxLength={40}/></>}<div className="record-status"><span className="sample-letter">{activeLabel || '?'}</span><div><strong>{selectedSign?.samples.length ?? 0} of 3 recordings</strong><p>{(selectedSign?.samples.length ?? 0) >= 3 ? 'Ready · add more examples for variety' : 'Three examples unlock recognition'}</p><div className="sample-dots">{Array.from({length:3},(_,i)=><span key={i} className={(selectedSign?.samples.length ?? 0) > i ? 'complete' : ''}/>)}</div></div></div><div className="record-instructions"><span className="instruction-number">1</span><p>Choose a label and form its ISL sign.</p><span className="instruction-number">2</span><p>After the countdown, hold or perform the sign for 1.8 seconds.</p><span className="instruction-number">3</span><p>Repeat with small natural variations in position and speed.</p></div><button className="primary-button record-button" disabled={!live || !!recording || (selectedSign?.samples.length ?? 0) >= MAX_SAMPLES} onClick={startRecording}>{recording ? <LoaderCircle size={17} className="spin"/> : <span className="record-dot"/>}{recording ? recordPhase : 'Record an example'}</button>{recording && <button className="cancel-record" onClick={() => setRecording(null)}>Cancel recording</button>}<p className="training-note">Landmark patterns are saved here, without camera images. This learns your examples; it does not verify that a sign is correct ISL.</p></div></section>}
          </div>
          <div className="below-workspace"><div className="how-card"><div className="mini-label">A LITTLE GUIDANCE GOES A LONG WAY</div><div className="steps-row"><div><span className="step-number">01</span><h3>Choose a mode</h3><p>Letters or word signs.<br/>No training needed.</p></div><ChevronRight size={17}/><div><span className="step-number">02</span><h3>Show your sign</h3><p>Hold a letter or capture<br/>one word sign at a time.</p></div><ChevronRight size={17}/><div><span className="step-number">03</span><h3>Be heard</h3><p>Build your message and<br/>let SignBridge speak it.</p></div></div></div><div className="demo-card"><span className="demo-icon"><Monitor size={21}/></span><h3>Curious? Take a quick look.</h3><p>Explore how signs become a message<br/>with a simulated demo.</p><button onClick={startDemo}>Try the demo <ArrowRight size={16}/></button></div></div>
          {savedPhrases.length > 0 && <section className="saved-messages"><h2><History size={18}/> Saved this session</h2><div>{savedPhrases.map((phrase,i)=><button key={i} onClick={() => setTranscript(phrase)}>{phrase}<ArrowRight size={14}/></button>)}</div></section>}
          {page === 'translate' && <div className="library-banner"><div><span className="library-banner-icon"><Layers3 size={19}/></span><p><strong>Built-in letters and word signs are ready to use.</strong><span>Choose Letters to fingerspell, or Words to recognize a short ISL sign. My signs uses your personal recordings.</span></p></div><button onClick={() => navigate('teach')}>Customize your signs <ArrowRight size={16}/></button></div>}
        </>}

        {page === 'library' && <><div className="library-actions"><div><span className="metric-count">{library.signs.length}</span> signs learned <span className="divider">/</span><span className="metric-count">{readySigns.length}</span> ready to recognize</div><div><button className="secondary-button" onClick={() => importRef.current?.click()}><Upload size={16}/> Import library</button><button className="secondary-button" disabled={!library.signs.length} onClick={() => download(JSON.stringify(library, null, 2), 'signbridge-isl-library.json', 'application/json')}><ArrowDownToLine size={16}/> Export backup</button><button className="primary-button" onClick={() => navigate('teach')}><Plus size={16}/> Teach a sign</button></div><input ref={importRef} type="file" accept="application/json,.json" hidden onChange={e => { void importLibrary(e.target.files?.[0]); e.target.value = ''; }}/></div>{!library.signs.length ? <div className="empty-library"><span><Layers3 size={35}/></span><h2>A space for the way you sign.</h2><p>No signs recorded yet. Start with a familiar letter or a short word,<br/>then record three examples to make it recognizable.</p><button className="primary-button" onClick={() => navigate('teach')}>Teach your first sign <ArrowRight size={16}/></button></div> : <div className="sign-grid">{library.signs.map(sign => <article key={sign.id} className="sign-card"><div className="sign-card-top"><span className="sign-kind">{sign.kind}</span><button aria-label={`Delete ${sign.label}`} title="Delete sign" onClick={() => setDeleteId(sign.id)}><Trash2 size={16}/></button></div><h2>{sign.label}</h2><p>{sign.samples.length} recordings <span>·</span> {sign.samples.length >= 3 ? 'Ready to recognize' : `${3 - sign.samples.length} more needed`}</p><button className="sign-practice" onClick={() => { setTrainingKind(sign.kind); if (sign.kind === 'letter') setTrainingLabel(sign.label); else setWordLabel(sign.label); navigate('teach'); }}>Add an example <ArrowRight size={15}/></button></article>)}</div>}<div className="backup-note"><ShieldCheck size={20}/><p><strong>Yours to keep. Yours to control.</strong><span>Your library is stored in this browser. Export a backup before clearing browser data or changing devices. Imports add new labels and keep existing recordings.</span></p></div></>}

        {page === 'guide' && <div className="guide-layout"><section className="guide-main"><h2>From a sign to a spoken message.</h2>{[{title:'Enable the camera and choose Letters',body:'Built-in ISL A–Z recognition needs no personal recordings. Hold a clear letter sign steadily to add it. Lower your hands briefly before repeating the same letter, and press Space to finish a word.'},{title:'Choose Words for whole-word signs',body:'Keep your face, shoulders and signing hands in view. Press Recognize a word, wait for the countdown, and perform one sign during the four-second capture. Supported word signs lists the pretrained vocabulary.'},{title:'Let your message be heard',body:'Review and correct the text when needed. Press Speak message, or enable automatic speech to hear completed words. Model scores are not a measure of real-world accuracy.'},{title:'Customize only when you need to',body:'Teach signs is optional. Record three examples of a personal letter or word, then choose My signs to use your own library. Your existing recordings and backups still work.'}].map((step,i)=><div className="guide-step" key={i}><span>{String(i+1).padStart(2,'0')}</span><div><h3>{step.title}</h3><p>{step.body}</p></div></div>)}<button className="primary-button" onClick={() => navigate('teach')}>Let’s teach your first sign <ArrowRight size={16}/></button></section><aside className="guide-aside"><Hand size={30}/><h3>Built-in ISL recognition.</h3><p>Letters uses the pretrained SignLens alphabet model. Words uses AI4Bharat’s INCLUDE word classifier. Both run locally; My signs uses your own recorded examples.</p><p>Built-in model scores and personal match scores do not measure validated accuracy. Similar poses, overlapping hands, signs outside the vocabulary and differences between signers can produce mistakes.</p><p>Use it for practice and demonstrations. Fluent, continuous ISL translation requires a validated dataset and a broader language model.</p><a href="https://islrtc.nic.in/document-category/awareness-material/" target="_blank" rel="noreferrer">Learn with official ISLRTC resources <ArrowRight size={15}/></a><div className="guide-privacy"><ShieldCheck size={18}/><span>Camera frames are processed locally. The app stores only hand landmarks. Browser speech services may use your device’s speech provider.</span></div></aside></div>}
        <footer><span><Hand size={14}/> A bridge to more connected conversations.</span><span>Made for your hands. Built around your voice.</span></footer>
      </main>
    </div>
    {deleteId && <div className="modal-backdrop"><div ref={modalRef} className="modal" role="dialog" aria-modal="true" aria-labelledby="delete-title"><button className="modal-close icon-button" onClick={() => setDeleteId(null)} aria-label="Cancel deletion"><X size={20}/></button><Trash2 size={24}/><h2 id="delete-title">Delete {library.signs.find(s=>s.id === deleteId)?.label}?</h2><p>This removes its recordings from this browser. Export a backup first if you want to keep them.</p><div className="modal-buttons"><button className="secondary-button" onClick={() => setDeleteId(null)}>Keep sign</button><button className="danger-button" onClick={() => { persist({version:1,signs:library.signs.filter(s=>s.id!==deleteId)}); setDeleteId(null); }}>Delete recordings</button></div></div></div>}
    {showSettings && <div className="modal-backdrop"><div ref={modalRef} className="modal" role="dialog" aria-modal="true" aria-labelledby="preferences-title"><button className="modal-close icon-button" onClick={() => setShowSettings(false)} aria-label="Close preferences"><X size={20}/></button><Settings2 size={24}/><h2 id="preferences-title">Make yourself comfortable.</h2><label className="toggle-row"><span className="toggle-title">Speak each detected letter</span><input type="checkbox" checked={speakLetters} onChange={e => setSpeakLetters(e.target.checked)}/><span className="switch"/></label><label className="toggle-row"><span className="toggle-title">Show hand landmarks</span><input type="checkbox" checked={camera.overlay} onChange={e => camera.setOverlay(e.target.checked)}/><span className="switch"/></label><p>Speech uses an available English voice on your device. Camera access stops when this tab is hidden.</p><button className="primary-button" onClick={() => setShowSettings(false)}><Check size={16}/> Done</button></div></div>}
  </div>;
}


