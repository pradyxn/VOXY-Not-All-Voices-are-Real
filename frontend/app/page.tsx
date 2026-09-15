"use client";

import { useEffect, useState } from "react";
import { AlertTriangle, ArrowRight, Check, FileAudio, LockKeyhole, Mic, Play, ShieldCheck, Upload, Waves, X } from "lucide-react";

const API = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

type Result = {
  status: string;
  synthetic_score: number;
  risk_score: number;
  risk_level: string;
  windows_analyzed: number;
  audio_quality: string;
  mode: string;
  model_message?: string;
  timeline: number[];
};

const demoResults: Record<string, Result> = {
  human: { status: "human", synthetic_score: 18, risk_score: 12, risk_level: "LOW", windows_analyzed: 4, audio_quality: "good", mode: "demo", model_message: "SIMULATION - deterministic demo output", timeline: [14, 21, 16, 18] },
  synthetic: { status: "suspicious", synthetic_score: 91, risk_score: 88, risk_level: "CRITICAL", windows_analyzed: 4, audio_quality: "good", mode: "demo", model_message: "SIMULATION - deterministic demo output", timeline: [84, 93, 89, 96] },
  uncertain: { status: "uncertain", synthetic_score: 52, risk_score: 43, risk_level: "MODERATE", windows_analyzed: 3, audio_quality: "good", mode: "demo", model_message: "SIMULATION - deterministic demo output", timeline: [45, 59, 51] },
};

const pipelineSteps = [
  ["VOICE", "The recording enters the same path whether it comes from a file or a microphone."],
  ["PREPROCESSING", "Audio is converted to 16 kHz mono before analysis."],
  ["4-SECOND WINDOWS", "Consistent windows keep the signal comparable as the voice changes."],
  ["MEL-SPECTROGRAM", "A 128-band Mel representation makes acoustic texture measurable."],
  ["NEURAL NETWORK", "The PyTorch CNN evaluates patterns associated with synthetic speech."],
  ["AGGREGATION", "Multiple windows are combined into a 0-100 risk signal."],
];

function scrollToSection(id: string) {
  document.getElementById(id)?.scrollIntoView({ behavior: "smooth", block: "start" });
}

export default function Home() {
  const [result, setResult] = useState<Result>(demoResults.uncertain);
  const [busy, setBusy] = useState(false);
  const [modal, setModal] = useState<"login" | "contact" | null>(null);
  const [callOpen, setCallOpen] = useState(false);
  const [verification, setVerification] = useState(false);
  const [activeStep, setActiveStep] = useState(0);

  useEffect(() => {
    const items = document.querySelectorAll(".reveal");
    const observer = new IntersectionObserver((entries) => entries.forEach((entry) => {
      if (entry.isIntersecting) entry.target.classList.add("is-visible");
    }), { threshold: 0.12 });
    items.forEach((item) => observer.observe(item));
    return () => observer.disconnect();
  }, []);

  async function analyzeFile(file?: File) {
    if (!file) return;
    setBusy(true);
    scrollToSection("analyze");
    try {
      const body = new FormData();
      body.append("audio", file);
      const response = await fetch(`${API}/api/analyze`, { method: "POST", body });
      if (!response.ok) throw new Error("Backend unavailable");
      setResult(await response.json());
    } catch {
      setResult({ ...demoResults.uncertain, model_message: "SIMULATION - backend unavailable; showing safe demo state" });
    } finally {
      setBusy(false);
    }
  }

  function chooseDemo(kind: keyof typeof demoResults) {
    setResult(demoResults[kind]);
    scrollToSection("analyze");
  }

  const title = busy ? "Analyzing..." : result.status === "suspicious" ? "Elevated signal" : result.status === "human" ? "Lower signal" : "Needs context";

  return (
    <main className="vxy-shell">
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
          <div className="hero-meta hero-meta-left"><span className="meta-label">VOICE ANALYSIS</span><strong>16 kHz</strong><span>4s WINDOWS</span><span>128-BAND MEL</span></div>
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
          <div className="analyze-layout"><div className="upload-card"><div className="card-title"><FileAudio size={20} /><h3>Voice input</h3></div><p className="muted">WAV, MP3, M4A, or browser WebM. Audio is processed temporarily.</p><label className="dropzone"><Upload size={27} /><strong>Choose a recording</strong><span>Send it to the VOXY pipeline</span><input type="file" accept="audio/*" onChange={(event) => analyzeFile(event.target.files?.[0])} /></label><div className="demo-row"><button onClick={() => chooseDemo("human")}>Human / demo</button><button onClick={() => chooseDemo("synthetic")}>Synthetic / demo</button><button onClick={() => chooseDemo("uncertain")}>Uncertain / demo</button></div><button className="text-action" onClick={() => setCallOpen(!callOpen)}><Mic size={16} /> {callOpen ? "Close realtime simulation" : "Open realtime call simulation"}</button>{callOpen && <div className="inline-note"><strong>Realtime Call Simulation</strong><p>A browser can analyze microphone input, but cannot intercept arbitrary cellular calls.</p><button className="vxy-pill primary" onClick={() => setCallOpen(false)}>Start simulated call</button></div>}</div>
            <div className="result-card" aria-live="polite"><div className="result-top"><div><span className="kicker">Voice assessment</span><h3>{title}</h3></div><span className="badge">{result.mode === "demo" ? "SIMULATION" : result.risk_level}</span></div><div className="score">{Math.round(result.synthetic_score)}<span>/100</span></div><p className="muted">synthetic likelihood · model score, not a calibrated probability</p><div className="meter"><i style={{ width: `${result.synthetic_score}%` }} /></div><div className="metrics"><div><small>Risk score</small><strong>{result.risk_score}</strong></div><div><small>Risk level</small><strong>{result.risk_level}</strong></div><div><small>Windows</small><strong>{result.windows_analyzed}</strong></div></div><div className="timeline-bars">{result.timeline.map((score, index) => <div key={index} style={{ height: `${Math.max(score, 8)}%` }} title={`Window ${index + 1}: ${score}`} />)}</div>{result.risk_score >= 60 && <div className="verify-note"><strong>Verify before taking action.</strong><p>Ask a trusted question, contact the person through another channel, and do not share OTPs or transfer money based on the call alone.</p><button className="vxy-pill primary" onClick={() => setVerification(!verification)}>{verification ? <><Check size={15} /> Steps shown</> : "Verify caller"}</button>{verification && <p><Check size={15} /> End the call, use a trusted contact method, and independently confirm identity.</p>}</div>}<p className="model-message">{result.model_message}</p></div></div>
        </section>

        <section className="vxy-section split-section reveal" id="insights"><div><span className="kicker">HOW VOXY WORKS</span><h2>From voice<br />to risk signal.</h2></div><p>Audio enters VOXY as a short voice sample. We normalize it to 16 kHz, split it into 4-second windows, convert each window into a Mel-spectrogram, and analyze it with our neural network. Results are aggregated across the recording to produce a 0–100 voice-risk score.</p></section>

        <footer className="vxy-footer"><span>VOXY · Not All Voices Are Real.</span><span><LockKeyhole size={14} /> Audio is processed for analysis and not intentionally retained.</span></footer>
      </div>
      {modal && <div className="modal-backdrop" role="dialog" aria-modal="true"><div className={`modal ${modal === "contact" ? "contact-modal" : ""}`}><button className="modal-close" onClick={() => setModal(null)} aria-label="Close"><X size={18} /></button>{modal === "login" ? <><span className="kicker">Private workspace</span><h2>Prototype access</h2><p>Authentication is not connected in this local prototype. Start with the live analysis workspace below.</p><button className="vxy-pill primary" onClick={() => { setModal(null); scrollToSection("analyze"); }}>Open workspace <ArrowRight size={15} /></button></> : <><span className="kicker">CONTACT</span><a className="contact-email" href="mailto:pradyun.marukala@gmail.com">pradyun.marukala@gmail.com</a></>}</div></div>}
    </main>
  );
}
