"use client";

import { useEffect, useRef, useState } from "react";
import { AlertTriangle, ArrowRight, Check, FileAudio, LockKeyhole, Mic, Moon, Play, ShieldCheck, Sun, Upload, Waves, X } from "lucide-react";
import { getWebSocketUrl } from "./websocket";

const API = process.env.NEXT_PUBLIC_API_URL?.trim() || (process.env.NODE_ENV === "development" ? "http://localhost:8000" : "");

function logWebSocket(message: string, details?: unknown) {
  if (process.env.NODE_ENV === "development") console.info(`[VOXY WebSocket] ${message}`, details ?? "");
}

type Result = {
  status: string;
  synthetic_score: number;
  risk_score: number;
  risk_level: string;
  windows_analyzed: number;
  audio_quality: string;
  mode: "pretrained" | "demo" | "live";
  detector?: string;
  model_message?: string;
  timeline: number[];
};

const demoResults: Record<string, Result> = {
  human: { status: "human", synthetic_score: 18, risk_score: 12, risk_level: "LOW", windows_analyzed: 4, audio_quality: "good", mode: "demo", model_message: "SIMULATION - deterministic demo output", timeline: [14, 21, 16, 18] },
  synthetic: { status: "suspicious", synthetic_score: 91, risk_score: 88, risk_level: "CRITICAL", windows_analyzed: 4, audio_quality: "good", mode: "demo", model_message: "SIMULATION - deterministic demo output", timeline: [84, 93, 89, 96] },
  uncertain: { status: "uncertain", synthetic_score: 52, risk_score: 43, risk_level: "MODERATE", windows_analyzed: 3, audio_quality: "good", mode: "demo", model_message: "SIMULATION - deterministic demo output", timeline: [45, 59, 51] },
};

const liveListeningResult: Result = {
  status: "uncertain",
  synthetic_score: 0,
  risk_score: 0,
  risk_level: "LOW",
  windows_analyzed: 0,
  audio_quality: "waiting",
  mode: "live",
  model_message: "LISTENING - waiting for the first complete voice window...",
  timeline: [],
};

const pipelineSteps = [
  ["VOICE", "The recording enters the same path whether it comes from a file or a microphone."],
  ["PREPROCESSING", "Audio is converted to 16 kHz mono before analysis."],
  ["4-SECOND WINDOWS", "Consistent windows keep the signal comparable as the voice changes."],
  ["RAW WAVEFORM", "A fixed 16 kHz waveform window enters the anti-spoofing model."],
  ["AASIST MODEL", "A pretrained graph-attention network evaluates bona fide and spoof cues."],
  ["AGGREGATION", "Multiple windows are combined into a 0-100 risk signal."],
];

function scrollToSection(id: string) {
  document.getElementById(id)?.scrollIntoView({ behavior: "smooth", block: "start" });
}

const LIVE_SAMPLE_RATE = 16000;
const LIVE_WINDOW_SECONDS = 4.0375;
const LIVE_TARGET_SAMPLES = 64600;

function downsampleToTarget(input: Float32Array): Float32Array {
  const output = new Float32Array(LIVE_TARGET_SAMPLES);
  if (!input.length) return output;
  const ratio = input.length / LIVE_TARGET_SAMPLES;
  for (let index = 0; index < LIVE_TARGET_SAMPLES; index += 1) {
    const position = index * ratio;
    const left = Math.min(Math.floor(position), input.length - 1);
    const right = Math.min(left + 1, input.length - 1);
    const fraction = position - left;
    output[index] = input[left] + (input[right] - input[left]) * fraction;
  }
  return output;
}

function encodeWav(samples: Float32Array, sampleRate: number): ArrayBuffer {
  const buffer = new ArrayBuffer(44 + samples.length * 2);
  const view = new DataView(buffer);
  const write = (offset: number, value: string) => {
    for (let index = 0; index < value.length; index += 1) view.setUint8(offset + index, value.charCodeAt(index));
  };
  write(0, 'RIFF'); write(8, 'WAVE'); write(12, 'fmt '); write(36, 'data');
  view.setUint32(4, 36 + samples.length * 2, true);
  view.setUint32(16, 16, true); view.setUint16(20, 1, true); view.setUint16(22, 1, true);
  view.setUint32(24, sampleRate, true); view.setUint32(28, sampleRate * 2, true);
  view.setUint16(32, 2, true); view.setUint16(34, 16, true); view.setUint32(40, samples.length * 2, true);
  for (let index = 0; index < samples.length; index += 1) {
    const sample = Math.max(-1, Math.min(1, samples[index]));
    view.setInt16(44 + index * 2, sample < 0 ? sample * 0x8000 : sample * 0x7fff, true);
  }
  return buffer;
}

export default function Home() {
  const [result, setResult] = useState<Result>(demoResults.uncertain);
  const [theme, setTheme] = useState<"light" | "dark">("light");
  const [busy, setBusy] = useState(false);
  const [analysisError, setAnalysisError] = useState<string | null>(null);
  const [modal, setModal] = useState<"login" | "contact" | null>(null);
  const [callOpen, setCallOpen] = useState(false);
  const [liveActive, setLiveActive] = useState(false);
  const [liveError, setLiveError] = useState<string | null>(null);
  const [liveSeconds, setLiveSeconds] = useState(0);
  const mediaRecorder = useRef<MediaRecorder | null>(null);
  const mediaStream = useRef<MediaStream | null>(null);
  const socket = useRef<WebSocket | null>(null);
  const intentionalStop = useRef(false);
  const stopSegments = useRef<(() => void) | null>(null);
  const [verification, setVerification] = useState(false);
  const audioContext = useRef<AudioContext | null>(null);
  const audioSource = useRef<MediaStreamAudioSourceNode | null>(null);
  const analyser = useRef<AnalyserNode | null>(null);
  const pcmProcessor = useRef<ScriptProcessorNode | null>(null);
  const silentGain = useRef<GainNode | null>(null);
  const waveformCanvas = useRef<HTMLCanvasElement | null>(null);
  const waveformFrame = useRef<number | null>(null);
  const snapshotTimer = useRef<number | null>(null);
  const pcmChunks = useRef<Float32Array[]>([]);
  const pcmSampleCount = useRef(0);
  const modelRequestPending = useRef(false);
  const [activeStep, setActiveStep] = useState(0);

  useEffect(() => {
    const savedTheme = window.localStorage.getItem("voxy-theme");
    if (savedTheme === "dark" || savedTheme === "light") setTheme(savedTheme);
  }, []);

  useEffect(() => {
    document.documentElement.dataset.theme = theme;
    window.localStorage.setItem("voxy-theme", theme);
  }, [theme]);

  useEffect(() => {
    if (!liveActive) return;
    const timer = window.setInterval(() => setLiveSeconds((seconds) => seconds + 1), 1000);
    return () => window.clearInterval(timer);
  }, [liveActive]);

  useEffect(() => () => stopLiveSimulation(), []);

  useEffect(() => {
    const items = document.querySelectorAll(".reveal");
    const observer = new IntersectionObserver((entries) => entries.forEach((entry) => {
      if (entry.isIntersecting) entry.target.classList.add("is-visible");
    }), { threshold: 0.12 });
    items.forEach((item) => observer.observe(item));
    return () => observer.disconnect();
  }, []);

  useEffect(() => {
    document.documentElement.style.colorScheme = theme;
    return () => {
      document.documentElement.style.colorScheme = "light";
    };
  }, [theme]);

  async function analyzeFile(file?: File) {
    if (!file) return;
    setBusy(true);
    setAnalysisError(null);
    scrollToSection("analyze");
    try {
      if (!API) throw new Error("Backend URL is missing. Set NEXT_PUBLIC_API_URL and rebuild the frontend.");
      const body = new FormData();
      body.append("audio", file);
      const response = await fetch(`${API}/api/analyze`, { method: "POST", body });
      if (!response.ok) {
        let message = `Audio analysis failed (${response.status}).`;
        try {
          const payload = await response.json() as { detail?: string };
          if (payload.detail) message = payload.detail;
        } catch {
          // Keep the status-based message when the server response is not JSON.
        }
        throw new Error(message);
      }
      setResult(await response.json());
    } catch (error) {
      setAnalysisError(error instanceof Error ? error.message : "Audio analysis failed.");
    } finally {
      setBusy(false);
    }
  }

  function chooseDemo(kind: keyof typeof demoResults) {
    stopLiveSimulation(true);
    setAnalysisError(null);
    setResult(demoResults[kind]);
    scrollToSection("analyze");
  }

  function stopLiveSimulation(silent = false) {
    intentionalStop.current = silent;
    if (snapshotTimer.current !== null) { window.clearInterval(snapshotTimer.current); snapshotTimer.current = null; }
    if (waveformFrame.current !== null) { window.cancelAnimationFrame(waveformFrame.current); waveformFrame.current = null; }
    if (pcmProcessor.current) { pcmProcessor.current.onaudioprocess = null; try { pcmProcessor.current.disconnect(); } catch {} pcmProcessor.current = null; }
    try { silentGain.current?.disconnect(); } catch {}
    try { audioSource.current?.disconnect(); } catch {}
    audioSource.current = null; analyser.current = null; silentGain.current = null;
    if (audioContext.current && audioContext.current.state !== 'closed') void audioContext.current.close();
    audioContext.current = null; pcmChunks.current = []; pcmSampleCount.current = 0; modelRequestPending.current = false;
    mediaStream.current?.getTracks().forEach((track) => track.stop());
    socket.current?.close(1000, 'client stopped');
    mediaStream.current = null; socket.current = null; setLiveActive(false);
  }

  async function startLiveSimulation() {
    setLiveError(null); setAnalysisError(null); setResult(liveListeningResult);
    pcmChunks.current = []; pcmSampleCount.current = 0; modelRequestPending.current = false;
    if (!navigator.mediaDevices?.getUserMedia || !window.AudioContext) { setLiveError('This browser does not support live microphone analysis.'); return; }
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const liveSocket = new WebSocket(getWebSocketUrl(API, window.location.protocol));
      const context = new AudioContext(); await context.resume();
      const source = context.createMediaStreamSource(stream);
      const analyserNode = context.createAnalyser(); analyserNode.fftSize = 2048; analyserNode.smoothingTimeConstant = 0.12;
      const processor = context.createScriptProcessor(4096, 1, 1); const silent = context.createGain(); silent.gain.value = 0;
      source.connect(analyserNode); source.connect(processor); processor.connect(silent); silent.connect(context.destination);
      mediaStream.current = stream; socket.current = liveSocket; audioContext.current = context; audioSource.current = source; analyser.current = analyserNode; pcmProcessor.current = processor; silentGain.current = silent; intentionalStop.current = false;

      const drawWaveform = () => {
        const canvas = waveformCanvas.current; const current = analyser.current;
        if (!canvas || !current) return;
        const dpr = window.devicePixelRatio || 1; const width = Math.max(1, Math.floor(canvas.clientWidth * dpr)); const height = Math.max(1, Math.floor(canvas.clientHeight * dpr));
        if (canvas.width !== width || canvas.height !== height) { canvas.width = width; canvas.height = height; }
        const ctx = canvas.getContext('2d'); if (!ctx) return;
        const data = new Uint8Array(current.fftSize); current.getByteTimeDomainData(data); ctx.clearRect(0, 0, width, height);
        ctx.strokeStyle = 'rgba(198,233,61,.95)'; ctx.lineWidth = Math.max(1.5, dpr * 1.5); ctx.beginPath();
        const dx = width / data.length;
        for (let index = 0; index < data.length; index += 1) { const x = index * dx; const y = height / 2 + ((data[index] - 128) / 128) * height * 0.42; if (index === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y); }
        ctx.stroke(); waveformFrame.current = window.requestAnimationFrame(drawWaveform);
      };

      processor.onaudioprocess = (event) => {
        const copy = new Float32Array(event.inputBuffer.getChannelData(0)); pcmChunks.current.push(copy); pcmSampleCount.current += copy.length;
        const keep = Math.ceil(context.sampleRate * (LIVE_WINDOW_SECONDS + 0.5));
        while (pcmChunks.current.length > 1 && pcmSampleCount.current > keep) { const removed = pcmChunks.current.shift(); if (removed) pcmSampleCount.current -= removed.length; }
      };

      const buildWindow = () => {
        const needed = Math.ceil(context.sampleRate * LIVE_WINDOW_SECONDS); if (pcmSampleCount.current < needed) return null;
        const joined = new Float32Array(pcmSampleCount.current); let offset = 0; for (const chunk of pcmChunks.current) { joined.set(chunk, offset); offset += chunk.length; }
        return downsampleToTarget(joined.subarray(Math.max(0, joined.length - needed)));
      };
      const sendWindow = () => {
        if (liveSocket.readyState !== WebSocket.OPEN || modelRequestPending.current) return;
        const samples = buildWindow(); if (!samples) return;
        modelRequestPending.current = true; liveSocket.send(encodeWav(samples, LIVE_SAMPLE_RATE));
      };

      liveSocket.onopen = () => { setLiveSeconds(0); setLiveActive(true); waveformFrame.current = window.requestAnimationFrame(drawWaveform); snapshotTimer.current = window.setInterval(sendWindow, 1000); };
      liveSocket.onmessage = (event) => {
        modelRequestPending.current = false;
        try {
          if (typeof event.data !== 'string') return;
          const update = JSON.parse(event.data) as Result & { error?: string };
          if (update.error) { setLiveError(update.error); return; }
          setResult({ ...update, mode: 'pretrained', model_message: 'LIVE MODEL ANALYSIS - rolling 4-second AASIST window' });
        } catch (error) { logWebSocket('invalid server message', error); }
      };
      liveSocket.onerror = () => logWebSocket('onerror', 'Network error or connection refused');
      liveSocket.onclose = (event) => {
        modelRequestPending.current = false;
        if (snapshotTimer.current !== null) { window.clearInterval(snapshotTimer.current); snapshotTimer.current = null; }
        if (waveformFrame.current !== null) { window.cancelAnimationFrame(waveformFrame.current); waveformFrame.current = null; }
        if (intentionalStop.current || event.code === 1000) setLiveError(null);
        else if (event.code === 1011 && event.reason.toLowerCase().includes('detector')) setLiveError('Backend closed the connection because the AASIST detector is unavailable.');
        else if (event.code === 1006) setLiveError('WebSocket connection refused or interrupted. Check that the backend is running and reachable.');
        else if (event.reason) setLiveError(`Backend closed the live connection (${event.code}): ${event.reason}`);
        else setLiveError(`Live connection closed unexpectedly (code ${event.code}).`);
        setLiveActive(false); mediaStream.current?.getTracks().forEach((track) => track.stop());
        if (audioContext.current && audioContext.current.state !== 'closed') void audioContext.current.close(); audioContext.current = null;
      };
    } catch (error) {
      setLiveError(error instanceof DOMException && error.name === 'NotAllowedError' ? 'Microphone permission was denied.' : error instanceof Error ? error.message : 'Unable to start microphone analysis.');
      stopLiveSimulation();
    }
  }

  const title = busy
    ? "Analyzing..."
    : liveActive && result.windows_analyzed === 0
      ? "Listening..."
      : result.status === "suspicious"
        ? "Elevated signal"
        : result.status === "human"
          ? "Lower signal"
          : "Needs context";

  const badge = liveActive
    ? result.windows_analyzed === 0
      ? "LISTENING"
      : "LIVE"
    : result.mode === "demo"
      ? "SIMULATION"
      : result.risk_level;

  return (
    <main className={`vxy-shell theme-${theme}`}>
      <style>{`
        .vxy-shell{--theme-transition:background .35s ease,color .35s ease,border-color .35s ease;transition:var(--theme-transition)}
        .vxy-shell.theme-dark{background:#050606;color:#f2f4ee}
        .vxy-shell.theme-dark .vxy-frame{background:#0d0f0e;color:#f2f4ee}
        .vxy-shell.theme-dark .vxy-nav{background:rgba(21,24,22,.76);border-color:rgba(255,255,255,.14)}
        .vxy-shell.theme-dark .vxy-logo,.vxy-shell.theme-dark .vxy-title,.vxy-shell.theme-dark .hero-meta,.vxy-shell.theme-dark .hero-meta-left strong,.vxy-shell.theme-dark .meta-label{color:#f2f4ee}
        .vxy-shell.theme-dark .vxy-nav-links button,.vxy-shell.theme-dark .hero-meta-right p,.vxy-shell.theme-dark .muted,.vxy-shell.theme-dark .kicker,.vxy-shell.theme-dark .split-section>p{color:#a8ada7}
        .vxy-shell.theme-dark .vxy-nav-links button:hover{color:#fff}
        .vxy-shell.theme-dark .vxy-pill.secondary,.vxy-shell.theme-dark .theme-toggle,.vxy-shell.theme-dark .pipeline-step,.vxy-shell.theme-dark .step-detail,.vxy-shell.theme-dark .upload-card,.vxy-shell.theme-dark .result-card,.vxy-shell.theme-dark .demo-row button,.vxy-shell.theme-dark .inline-note,.vxy-shell.theme-dark .verify-note,.vxy-shell.theme-dark .metrics>div,.vxy-shell.theme-dark .dropzone{background:#151816;color:#f2f4ee;border-color:rgba(255,255,255,.14)}
        .vxy-shell.theme-dark .pipeline-step:hover,.vxy-shell.theme-dark .pipeline-step.active{background:#1b1e1c;border-color:#c6e93d}
        .vxy-shell.theme-dark .step-detail p,.vxy-shell.theme-dark .model-message{color:#a8ada7}
        .vxy-shell.theme-dark .dropzone strong,.vxy-shell.theme-dark .card-title h3,.vxy-shell.theme-dark .result-top h3{color:#f2f4ee}
        .vxy-shell.theme-dark .dropzone:hover{background:#1b1e1c}
        .vxy-shell.theme-dark .badge{background:#242824;color:#f2f4ee}
        .vxy-shell.theme-dark .meter,.vxy-shell.theme-dark .vxy-progress-bar{background:#292d2a}
        .vxy-shell.theme-dark .vxy-footer{border-color:rgba(255,255,255,.12)}
        .vxy-shell.theme-dark .contact-email{color:#f2f4ee}
        .theme-toggle{white-space:nowrap}
        .vxy-shell .vxy-nav-links button,.vxy-shell .vxy-pill,.vxy-shell .theme-toggle,.vxy-shell .vxy-logo,.vxy-shell .demo-row button,.vxy-shell .text-action,.vxy-shell .contact-email{transition:transform .2s ease,color .2s ease,background .2s ease,border-color .2s ease}
        @media (hover:hover) and (pointer:fine){
          .vxy-shell .vxy-nav-links button:hover,.vxy-shell .text-action:hover,.vxy-shell .contact-email:hover{transform:scale(1.035) translateY(-1px)}
          .vxy-shell .vxy-pill:hover,.vxy-shell .theme-toggle:hover{transform:scale(1.04) translateY(-2px)}
          .vxy-shell .vxy-pill:active,.vxy-shell .theme-toggle:active{transform:scale(.98)}
          .vxy-shell .demo-row button:hover{transform:scale(1.03) translateY(-1px)}
        }
        @media (prefers-reduced-motion:reduce){.vxy-shell .vxy-nav-links button,.vxy-shell .vxy-pill,.vxy-shell .theme-toggle,.vxy-shell .vxy-logo,.vxy-shell .demo-row button,.vxy-shell .text-action,.vxy-shell .contact-email{transition:none}}
      `}</style>
      <div className="vxy-frame">
        <header className="vxy-header">
          <nav className="vxy-nav" aria-label="Main navigation">
            <button className="vxy-logo" onClick={() => scrollToSection("top")} aria-label="Back to top">VOXY<span>.</span></button>
            <div className="vxy-nav-links">
              <button onClick={() => scrollToSection("analyze")}>Services</button>
              <button onClick={() => scrollToSection("about")}>About</button>
              <button onClick={() => scrollToSection("insights")}>Insights</button>
              <button onClick={() => setModal("contact")}>Contact</button>
            </div>
            <div className="vxy-nav-actions">
              <button className="theme-toggle" type="button" onClick={() => setTheme(theme === "light" ? "dark" : "light")} aria-label={`Switch to ${theme === "light" ? "dark" : "light"} mode`} aria-pressed={theme === "dark"} title={`Switch to ${theme === "light" ? "dark" : "light"} mode`}>
                {theme === "light" ? <Moon size={16} /> : <Sun size={16} />}
                <span>{theme === "light" ? "Dark" : "Light"}</span>
              </button>
              <button className="vxy-pill secondary" onClick={() => setModal("login")}>Login</button>
              <button className="vxy-pill primary" onClick={() => scrollToSection("analyze")}>Get Started <ArrowRight size={15} /></button>
            </div>
          </nav>
        </header>

        <section className="vxy-hero" id="top">
          <div className="vxy-title-wrap"><p className="vxy-eyebrow">VOICE INTELLIGENCE / 2026</p><h1 className="vxy-title"><span>NOT ALL</span><span>VOICES</span><span>ARE</span><span className="title-final">REAL<em>.</em></span></h1></div>
          <div className="vxy-orb-scene" aria-label="Abstract VOXY acoustic analysis visual">
            <div className="vxy-acoustic-visual"><div className="acoustic-halo halo-one" /><div className="acoustic-halo halo-two" /><div className="acoustic-core"><i /><i /><i /><i /></div><div className="acoustic-ribbon ribbon-one" /><div className="acoustic-ribbon ribbon-two" /><div className="acoustic-ribbon ribbon-three" /><div className="acoustic-spark spark-one" /><div className="acoustic-spark spark-two" /></div>
          </div>
          <div className="hero-meta hero-meta-left"><span className="meta-label">VOICE ANALYSIS</span><strong>16 kHz</strong><span>4s WINDOWS</span><span>RAW WAVEFORM</span></div>
          <div className="hero-meta hero-meta-right"><div className="process-line"><span>VOICE INPUT</span><small>/01</small></div><div className="process-line"><span>SIGNAL ANALYSIS</span><small>/02</small></div><div className="process-line"><span>RISK AGGREGATION</span><small>/03</small></div><p>Turn acoustic evidence<br />into your next safe step.</p></div>
          <div className="vxy-cta-wrap"><button className="vxy-cta-bubble" onClick={() => scrollToSection("analyze")} aria-label="Analyze a voice"><Play size={14} fill="currentColor" /><span>Analyze<br />a voice</span></button></div>
        </section>
        <div className="vxy-progress-bar"><div className="vxy-progress-handle" /></div>

        <section className="vxy-section dark reveal" id="about">
          <div className="dark-intro"><h2><span>Everything you</span><span>need</span><span>to question a</span><mark>voice.</mark></h2><p>VOXY helps you examine acoustic evidence associated with synthetic or cloned speech before you trust the call.</p></div>
          <div className="feature-grid"><article><div className="feature-icon"><Mic /></div><span className="card-number">01</span><h3>VOICE INPUT</h3><p>Upload a voice recording and send it through the VOXY analysis pipeline.</p></article><article><div className="feature-icon"><Waves /></div><span className="card-number">02</span><h3>SIGNAL ANALYSIS</h3><p>Audio is normalized and converted into Mel-spectrogram features for model analysis.</p></article><article><div className="feature-icon"><AlertTriangle /></div><span className="card-number">03</span><h3>AI DETECTION</h3><p>Our neural network evaluates acoustic patterns associated with synthetic or cloned speech.</p></article><article><div className="feature-icon"><ShieldCheck /></div><span className="card-number">04</span><h3>RISK SIGNAL</h3><p>Results are aggregated into a 0-100 risk score to help you decide what to do next.</p></article></div>
        </section>

        <section className="vxy-section reveal" id="works">
          <span className="kicker">Detection pipeline</span><h2>Voice into evidence.</h2>
          <div className="pipeline">{pipelineSteps.map(([step], index) => <button className={`pipeline-step ${activeStep === index ? "active" : ""}`} key={step} onClick={() => setActiveStep(index)}><small>0{index + 1}</small><strong>{step}</strong></button>)}</div>
          <div className="step-detail"><span>Step 0{activeStep + 1}</span><p>{pipelineSteps[activeStep][1]}</p></div>
        </section>

        <section className="vxy-section analyze-section reveal" id="analyze">
          <div className="section-heading"><div><span className="kicker">Live workspace</span><h2>Analyze a voice.</h2></div><p>Upload an audio clip or choose a labelled simulation. Demo values are never presented as genuine model predictions.</p></div>
          <div className="analyze-layout"><div className="upload-card"><div className="card-title"><FileAudio size={20} /><h3>Voice input</h3></div><p className="muted">WAV, MP3, M4A, or browser WebM. Audio is processed temporarily.</p><label className="dropzone"><Upload size={27} /><strong>Choose a recording</strong><span>Send it to the VOXY pipeline</span><input type="file" accept="audio/*" onChange={(event) => analyzeFile(event.target.files?.[0])} /></label><div className="demo-row"><button onClick={() => chooseDemo("human")}>Human / demo</button><button onClick={() => chooseDemo("synthetic")}>Synthetic / demo</button><button onClick={() => chooseDemo("uncertain")}>Uncertain / demo</button></div><button className="text-action" onClick={() => { if (liveActive) stopLiveSimulation(true); else setCallOpen(!callOpen); }}><Mic size={16} /> {liveActive ? "Stop microphone simulation" : callOpen ? "Close microphone simulation" : "Open microphone simulation"}</button>{callOpen && <div className="inline-note"><strong>Browser Microphone Simulation</strong><p>Audio is streamed to VOXY for live AASIST analysis. It does not intercept cellular calls.</p>{liveActive ? <p className="live-status"><span className="live-dot" /> Listening · {liveSeconds}s</p> : <button className="vxy-pill primary" onClick={startLiveSimulation}>Start microphone analysis</button>}{liveError && <p role="alert">{liveError}</p>}</div>}</div>
            <div className="result-card" aria-live="polite"><div className="result-top"><div><span className="kicker">Voice assessment</span><h3>{title}</h3></div><span className="badge">{badge}</span></div><div className="score">{liveActive && result.windows_analyzed === 0 ? "—" : Math.round(result.synthetic_score)}<span>/100</span></div><p className="muted">synthetic likelihood · model score, not a calibrated probability</p>{result.detector && result.mode !== "demo" && <p className="detector-label">{result.detector}</p>}<div className="meter"><i style={{ width: `${Math.max(0, Math.min(100, result.synthetic_score))}%` }} /></div><div className="metrics"><div><small>Risk score</small><strong>{result.risk_score}</strong></div><div><small>Risk level</small><strong>{result.risk_level}</strong></div><div><small>Windows</small><strong>{result.windows_analyzed}</strong></div></div><div className="timeline-graph" aria-label="Live synthetic score timeline">
              <svg viewBox="0 0 400 130" role="img" aria-label="Synthetic score by analysis window" style={{ width: "100%", height: "170px", display: "block", color: "#c6e93d" }}>
                <line x1="0" y1="20" x2="400" y2="20" stroke="currentColor" strokeOpacity=".12" />
                <line x1="0" y1="65" x2="400" y2="65" stroke="currentColor" strokeOpacity=".12" />
                <line x1="0" y1="110" x2="400" y2="110" stroke="currentColor" strokeOpacity=".12" />
                {result.timeline.length > 0 && (() => {
                  const denominator = Math.max(result.timeline.length - 1, 1);
                  const points = result.timeline.map((score, index) => {
                    const x = (index / denominator) * 400;
                    const y = 110 - (Math.max(0, Math.min(score, 100)) * 0.9);
                    return { x, y, score, index };
                  });
                  return (
                    <>
                      <polyline fill="none" stroke="currentColor" strokeWidth="3" strokeLinecap="round" strokeLinejoin="round" points={points.map((point) => `${point.x},${point.y}`).join(" ")} />
                      {points.map((point) => (
                        <circle key={point.index} cx={point.x} cy={point.y} r="4.5" fill="currentColor">
                          <title>{`Window ${point.index + 1}: ${point.score}`}</title>
                        </circle>
                      ))}
                    </>
                  );
                })()}
              </svg>
            </div>
            <div className="live-waveform-panel" style={{ marginTop: '10px', borderTop: '1px solid rgba(198,233,61,.12)', paddingTop: '10px' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '7px', fontSize: '10px', letterSpacing: '.12em', textTransform: 'uppercase', color: 'rgba(198,233,61,.78)' }}>
                <span>Live audio waveform</span><span>{liveActive ? 'MIC INPUT' : 'STANDBY'}</span>
              </div>
              <canvas ref={waveformCanvas} aria-label="Live microphone waveform" style={{ width: '100%', height: '105px', display: 'block', borderRadius: '10px', background: 'rgba(255,255,255,.018)' }} />
            </div>{result.risk_score >= 60 && <div className="verify-note"><strong>Verify before taking action.</strong><p>Ask a trusted question, contact the person through another channel, and do not share OTPs or transfer money based on the call alone.</p><button className="vxy-pill primary" onClick={() => setVerification(!verification)}>{verification ? <><Check size={15} /> Steps shown</> : "Verify caller"}</button>{verification && <p><Check size={15} /> End the call, use a trusted contact method, and independently confirm identity.</p>}</div>}<p className="model-message">{analysisError || result.model_message}</p></div></div>
        </section>

        <section className="vxy-section split-section reveal" id="insights"><div><span className="kicker">HOW VOXY WORKS</span><h2>From voice<br />to risk signal.</h2></div><p>Audio enters VOXY as a short voice sample. We normalize it to 16 kHz, split it into overlapping 4-second windows, and analyze each raw waveform with the pretrained AASIST anti-spoofing model. Results are aggregated across the recording to produce a 0–100 spoof-risk score.</p></section>

        <footer className="vxy-footer"><span>VOXY · Not All Voices Are Real.</span><span><LockKeyhole size={14} /> Audio is processed for analysis and not intentionally retained.</span></footer>
      </div>
      {modal && <div className="modal-backdrop" role="dialog" aria-modal="true"><div className={`modal ${modal === "contact" ? "contact-modal" : ""}`}><button className="modal-close" onClick={() => setModal(null)} aria-label="Close"><X size={18} /></button>{modal === "login" ? <><span className="kicker">Private workspace</span><h2>Prototype access</h2><p>Authentication is not connected in this local prototype. Start with the live analysis workspace below.</p><button className="vxy-pill primary" onClick={() => { setModal(null); scrollToSection("analyze"); }}>Open workspace <ArrowRight size={15} /></button></> : <><span className="kicker">CONTACT</span><a className="contact-email" href="mailto:pradyun.marukala@gmail.com">pradyun.marukala@gmail.com</a></>}</div></div>}
    </main>
  );
}
