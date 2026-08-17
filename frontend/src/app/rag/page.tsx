"use client";

import React, { useState, useRef } from "react";
import Link from "next/link";
import Image from "next/image";
import GoaMapBackground from "../components/GoaMapBackground";
import VoiceRecorder, { PipelineStage } from "../components/VoiceRecorder";
import PipelineVisualizer from "../components/PipelineVisualizer";
import AnswerCard, { AnswerData } from "../components/AnswerCard";
import TelemetryPanel from "../components/TelemetryPanel";
import "../globals.css";

// @ts-ignore
const API_BASE = (typeof import.meta !== "undefined" && (import.meta as any).env?.VITE_API_BASE_URL) || process.env.NEXT_PUBLIC_API_URL || process.env.VITE_API_BASE_URL || "";

interface VoiceMetadata {
  transcriptText?: string;
  sttProvider?: string;
  sttLanguage?: string;
  sttFallbackUsed?: boolean;
  sttLatencyMs?: number;
}

export default function RAGPage() {
  const [stage, setStage] = useState<PipelineStage>("IDLE");
  const [isRecording, setIsRecording] = useState(false);
  const [answerData, setAnswerData] = useState<AnswerData | null>(null);
  const [latencyMetrics, setLatencyMetrics] = useState<Record<string, number> | undefined>();
  const [textInput, setTextInput] = useState("");
  const [errorMessage, setErrorMessage] = useState<string | null>(null);

  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const chunksRef = useRef<BlobPart[]>([]);
  const currentRequestIdRef = useRef<number>(0);
  const requestIdCounter = useRef<number>(0);

  // --- Voice Recording Lifecycle ---
  const startRecording = async () => {
    setErrorMessage(null);
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      
      let mimeType = 'audio/webm';
      if (typeof MediaRecorder !== 'undefined' && !MediaRecorder.isTypeSupported(mimeType)) {
        if (MediaRecorder.isTypeSupported('audio/mp4')) {
          mimeType = 'audio/mp4';
        } else if (MediaRecorder.isTypeSupported('audio/ogg')) {
          mimeType = 'audio/ogg';
        }
      }

      const options = MediaRecorder.isTypeSupported(mimeType) ? { mimeType } : undefined;
      const mediaRecorder = new MediaRecorder(stream, options);
      mediaRecorderRef.current = mediaRecorder;
      chunksRef.current = [];

      mediaRecorder.ondataavailable = (e) => {
        if (e.data.size > 0) {
          chunksRef.current.push(e.data);
        }
      };

      mediaRecorder.onstop = async () => {
        const actualMime = mediaRecorder.mimeType || mimeType;
        const audioBlob = new Blob(chunksRef.current, { type: actualMime });
        stream.getTracks().forEach((track) => track.stop());
        await processAudio(audioBlob, actualMime);
      };

      mediaRecorder.start();
      setIsRecording(true);
      setStage("LISTENING");
    } catch (err: unknown) {
      console.error("Microphone error:", err);
      setErrorMessage("Microphone access is required. Please allow microphone permissions.");
      setStage("ERROR");
    }
  };

  const stopRecording = () => {
    if (mediaRecorderRef.current && isRecording) {
      mediaRecorderRef.current.stop();
      mediaRecorderRef.current.stream.getTracks().forEach((track) => track.stop());
      setIsRecording(false);
      setStage("PROCESSING");
    }
  };

  const cancelRecording = () => {
    if (mediaRecorderRef.current && isRecording) {
      mediaRecorderRef.current.onstop = null;
      mediaRecorderRef.current.stop();
      mediaRecorderRef.current.stream.getTracks().forEach((track) => track.stop());
      setIsRecording(false);
      setStage("IDLE");
    }
  };

  // --- Voice Transcription & RAG Execution ---
  const processAudio = async (audioBlob: Blob, mimeType: string) => {
    const requestId = ++requestIdCounter.current;
    currentRequestIdRef.current = requestId;
    setStage("TRANSCRIBING");

    const formData = new FormData();
    const ext = mimeType.includes("mp4") ? "m4a" : mimeType.includes("ogg") ? "ogg" : "webm";
    formData.append("audio", audioBlob, `recording.${ext}`);

    try {
      setTimeout(() => {
        if (currentRequestIdRef.current === requestId && stage !== "ERROR") {
          setStage("RETRIEVING");
        }
      }, 500);

      setTimeout(() => {
        if (currentRequestIdRef.current === requestId && stage !== "ERROR") {
          setStage("GENERATING");
        }
      }, 1000);

      const response = await fetch(`${API_BASE}/api/voice_ask`, {
        method: "POST",
        body: formData,
      });

      if (currentRequestIdRef.current !== requestId) return;

      if (!response.ok) {
        throw new Error(`Server returned HTTP ${response.status}`);
      }

      const data = await response.json();

      if (data.status === "error" || (data.transcription && !data.transcription.success)) {
        setErrorMessage(data.transcription?.error || data.message || "Could not transcribe audio.");
        setStage("ERROR");
        return;
      }

      const transcription = data.transcription || {};
      const transcript = transcription.text || data.query || "";

      setLatencyMetrics(data.latency_metrics);

      setAnswerData({
        query: transcript,
        transcriptText: transcript,
        sttProvider: transcription.provider,
        sttLanguage: transcription.language || data.routing?.language,
        sttFallbackUsed: transcription.fallback_used,
        answer: data.answer || data.message,
        grounded: data.grounded,
        contextSufficient: data.context_sufficient,
        refusalReason: data.refusal_reason,
        sources: data.sources || [],
        retrievalConfidence: data.confidence || data.retrieval_confidence,
      });

      setStage("ANSWER");
    } catch (err: unknown) {
      if (currentRequestIdRef.current !== requestId) return;
      console.error("Voice ask error:", err);
      setErrorMessage("Network error connecting to voice service. Please ensure backend is running.");
      setStage("ERROR");
    }
  };

  // --- RAG Query Execution ---
  const executeRAGQuery = async (queryText: string, voiceMetadata?: VoiceMetadata) => {
    const requestId = ++requestIdCounter.current;
    currentRequestIdRef.current = requestId;

    if (!voiceMetadata) {
      setStage("RETRIEVING");
    }

    try {
      setTimeout(() => {
        if (currentRequestIdRef.current === requestId && stage !== "ERROR") {
          setStage("GENERATING");
        }
      }, 400);

      const response = await fetch(`${API_BASE}/api/ask`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query: queryText }),
      });

      if (currentRequestIdRef.current !== requestId) return;

      if (!response.ok) {
        throw new Error(`Server returned HTTP ${response.status}`);
      }

      const ragResponse = await response.json();

      const combinedMetrics = {
        ...(ragResponse.latency_metrics || {}),
        stt_ms: voiceMetadata?.sttLatencyMs || ragResponse.latency_metrics?.stt_ms,
      };
      setLatencyMetrics(combinedMetrics);

      setAnswerData({
        query: queryText,
        transcriptText: voiceMetadata?.transcriptText || queryText,
        sttProvider: voiceMetadata?.sttProvider,
        sttLanguage: voiceMetadata?.sttLanguage || ragResponse.routing?.language,
        sttFallbackUsed: voiceMetadata?.sttFallbackUsed,
        answer: ragResponse.answer || ragResponse.message,
        grounded: ragResponse.grounded,
        contextSufficient: ragResponse.context_sufficient,
        refusalReason: ragResponse.refusal_reason,
        sources: ragResponse.sources || [],
        retrievalConfidence: ragResponse.retrieval_confidence,
      });

      setStage("ANSWER");
    } catch (err: unknown) {
      if (currentRequestIdRef.current !== requestId) return;
      console.error("RAG error:", err);
      setErrorMessage("Could not retrieve answer. Please try again.");
      setStage("ERROR");
    }
  };

  const handleTextSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!textInput.trim() || ["LISTENING", "PROCESSING", "TRANSCRIBING", "RETRIEVING", "GENERATING"].includes(stage)) return;
    const q = textInput.trim();
    setTextInput("");
    executeRAGQuery(q);
  };

  const quickPrompts = [
    { label: "What is the capital of India?", lang: "EN" },
    { label: "Who leads India?", lang: "EN" },
    { label: "भारत की राजधानी क्या है?", lang: "HI" },
    { label: "महाराष्ट्राची राजधानी कोणती आहे?", lang: "MR" },
  ];

  return (
    <div style={{ position: "relative", minHeight: "100vh", display: "flex", flexDirection: "column", backgroundColor: "var(--bg-primary)" }}>
      {/* 1. Subtle Digital Goa Map Background Layer */}
      <GoaMapBackground />

      {/* 2. Floating Glass Navigation Bar */}
      <header
        style={{
          position: "sticky",
          top: "16px",
          zIndex: 30,
          maxWidth: "1080px",
          width: "calc(100% - 32px)",
          margin: "0 auto",
          backdropFilter: "blur(20px)",
          WebkitBackdropFilter: "blur(20px)",
          border: "1px solid rgba(255, 255, 255, 0.08)",
          background: "rgba(13, 17, 24, 0.85)",
          borderRadius: "9999px",
          padding: "10px 22px",
          boxShadow: "0 10px 30px rgba(0, 0, 0, 0.5), 0 0 0 1px rgba(255, 255, 255, 0.04)",
        }}
      >
        <div
          style={{
            display: "flex",
            alignItems: "center",
            justifyContent: "space-between",
            flexWrap: "wrap",
            gap: "12px",
          }}
        >
          {/* Brand Logo & Return Link */}
          <div style={{ display: "flex", alignItems: "center", gap: "12px" }}>
            <Link
              href="/"
              className="btn-secondary"
              style={{
                padding: "6px 12px",
                fontSize: "0.78rem",
                textDecoration: "none",
                gap: "5px",
              }}
            >
              ← Home
            </Link>

            <div style={{ display: "flex", alignItems: "center", gap: "10px" }}>
              <div
                style={{
                  width: "34px",
                  height: "34px",
                  borderRadius: "10px",
                  background: "linear-gradient(135deg, rgba(0, 168, 120, 0.25) 0%, rgba(22, 138, 173, 0.2) 100%)",
                  border: "1px solid rgba(0, 168, 120, 0.4)",
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "center",
                }}
              >
                <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#16C784" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round">
                  <path d="M12 2v20M17 5v14M7 9v6M22 10v4M2 11v2" />
                </svg>
              </div>

              <div style={{ display: "flex", flexDirection: "column" }}>
                <div style={{ display: "flex", alignItems: "center", gap: "8px" }}>
                  <span className="font-display" style={{ fontSize: "0.95rem", fontWeight: 800, color: "#F5F7FA" }}>
                    Voice RAG Goa
                  </span>
                  <span className="badge badge-node" style={{ fontSize: "0.62rem", padding: "1px 6px" }}>
                    15.60° N • VAGATOR
                  </span>
                </div>
              </div>
            </div>
          </div>

          {/* Languages & Live Status */}
          <div style={{ display: "flex", alignItems: "center", gap: "10px", flexWrap: "wrap" }}>
            <div className="badge badge-lang" style={{ gap: "8px", padding: "5px 12px" }}>
              <span style={{ color: "var(--tropical-green)", fontWeight: 700 }}>EN</span> • <span>हिन्दी</span> • <span>मराठी</span> • <span>कोंकणी</span>
            </div>
            <span className="badge badge-live">
              <span style={{ width: "6px", height: "6px", borderRadius: "50%", background: "#16C784", animation: "pulseDot 1.5s infinite" }} />
              Live AI
            </span>
          </div>
        </div>
      </header>

      {/* 3. Main Viewport Content */}
      <main
        style={{
          position: "relative",
          zIndex: 10,
          flex: 1,
          maxWidth: "880px",
          width: "100%",
          margin: "0 auto",
          padding: "36px 20px 60px 20px",
          display: "flex",
          flexDirection: "column",
          gap: "28px",
        }}
      >
        {/* Hero Section */}
        <section style={{ textAlign: "center", display: "flex", flexDirection: "column", gap: "10px" }}>
          <div style={{ display: "inline-flex", justifyContent: "center" }}>
            <span
              style={{
                fontSize: "0.72rem",
                fontWeight: 800,
                letterSpacing: "0.14em",
                textTransform: "uppercase",
                color: "var(--tropical-green)",
                background: "rgba(0, 168, 120, 0.1)",
                border: "1px solid rgba(0, 168, 120, 0.3)",
                padding: "4px 14px",
                borderRadius: "9999px",
                display: "inline-flex",
                alignItems: "center",
                gap: "6px",
              }}
            >
              <span>🌴</span> Hacker House Goa 2026 • Task 2
            </span>
          </div>

          <h1
            className="font-display gradient-title"
            style={{
              fontSize: "clamp(2.3rem, 5.5vw, 3.6rem)",
              lineHeight: 1.12,
              fontWeight: 800,
              letterSpacing: "-0.03em",
            }}
          >
            Voice-Enabled <span className="gradient-goa-accent">RAG</span>
          </h1>

          <p style={{ fontSize: "1.05rem", color: "var(--text-secondary)", maxWidth: "580px", margin: "0 auto", lineHeight: 1.6 }}>
            Speak your question in English, Hindi, Marathi, or Konkani. Retrieve accurate knowledge with calibrated ground truth verification.
          </p>
        </section>

        {/* Centerpiece Microphone Voice Interaction Panel */}
        <section className="glass-panel-elevated" style={{ padding: "32px 24px 28px 24px" }}>
          <VoiceRecorder
            stage={stage}
            isRecording={isRecording}
            onStartRecording={startRecording}
            onStopRecording={stopRecording}
            onCancelRecording={cancelRecording}
            errorMessage={errorMessage}
          />

          {/* Quick Search Input Form */}
          <form
            onSubmit={handleTextSubmit}
            style={{
              marginTop: "22px",
              maxWidth: "620px",
              margin: "22px auto 0 auto",
              display: "flex",
              gap: "8px",
            }}
          >
            <input
              type="text"
              placeholder="Or type a question (English, हिन्दी, मराठी, कोंकणी)..."
              value={textInput}
              onChange={(e) => setTextInput(e.target.value)}
              style={{
                flex: 1,
                background: "rgba(8, 10, 15, 0.75)",
                border: "1px solid rgba(255, 255, 255, 0.1)",
                borderRadius: "12px",
                padding: "12px 16px",
                color: "#F5F7FA",
                fontSize: "0.95rem",
                outline: "none",
                transition: "border-color 0.2s ease, box-shadow 0.2s ease",
              }}
              onFocus={(e) => {
                e.target.style.borderColor = "var(--goa-green)";
                e.target.style.boxShadow = "0 0 14px rgba(0, 168, 120, 0.25)";
              }}
              onBlur={(e) => {
                e.target.style.borderColor = "rgba(255, 255, 255, 0.1)";
                e.target.style.boxShadow = "none";
              }}
            />
            <button
              type="submit"
              disabled={!textInput.trim()}
              className="btn-primary"
              style={{
                opacity: textInput.trim() ? 1 : 0.45,
                cursor: textInput.trim() ? "pointer" : "default",
                padding: "0 22px",
                borderRadius: "12px",
              }}
            >
              Ask
            </button>
          </form>

          {/* Quick 1-Click Suggestion Chips */}
          <div
            style={{
              marginTop: "16px",
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              flexWrap: "wrap",
              gap: "8px",
            }}
          >
            <span style={{ fontSize: "0.75rem", color: "var(--text-muted)", fontWeight: 600 }}>Try:</span>
            {quickPrompts.map((p, idx) => (
              <button
                key={idx}
                type="button"
                onClick={() => {
                  setTextInput(p.label);
                  executeRAGQuery(p.label);
                }}
                style={{
                  background: "rgba(255, 255, 255, 0.03)",
                  border: "1px solid rgba(255, 255, 255, 0.08)",
                  borderRadius: "9999px",
                  padding: "4px 10px",
                  color: "var(--text-secondary)",
                  fontSize: "0.76rem",
                  cursor: "pointer",
                  transition: "all 0.18s ease",
                }}
                onMouseEnter={(e) => {
                  e.currentTarget.style.borderColor = "var(--goa-green)";
                  e.currentTarget.style.color = "var(--text-primary)";
                  e.currentTarget.style.background = "rgba(0, 168, 120, 0.08)";
                }}
                onMouseLeave={(e) => {
                  e.currentTarget.style.borderColor = "rgba(255, 255, 255, 0.08)";
                  e.currentTarget.style.color = "var(--text-secondary)";
                  e.currentTarget.style.background = "rgba(255, 255, 255, 0.03)";
                }}
              >
                {p.label}
              </button>
            ))}
          </div>
        </section>

        {/* Real-time Pipeline Visualizer */}
        <PipelineVisualizer
          currentStage={stage}
          latencies={{
            stt_ms: latencyMetrics?.stt_ms,
            retrieval_ms: (latencyMetrics?.dense_retrieval_ms || 0) + (latencyMetrics?.sparse_retrieval_ms || 0) + (latencyMetrics?.rrf_fusion_ms || 0),
            generation_ms: latencyMetrics?.generation_ms,
            total_ms: latencyMetrics?.total_e2e_ms,
          }}
        />

        {/* Answer Experience & Grounded Passages */}
        {answerData && (
          <AnswerCard
            data={answerData}
            onEditQuery={(editedQ) => executeRAGQuery(editedQ)}
          />
        )}

        {/* Performance & Real Telemetry Panel */}
        <TelemetryPanel metrics={latencyMetrics} />
      </main>

      {/* 4. Minimal Goa Nightlife Footer */}
      <footer
        style={{
          borderTop: "1px solid var(--border-subtle)",
          padding: "24px 20px",
          textAlign: "center",
          color: "var(--text-muted)",
          fontSize: "0.82rem",
          display: "flex",
          flexDirection: "column",
          alignItems: "center",
          gap: "8px",
          background: "rgba(8, 10, 15, 0.8)",
          position: "relative",
          zIndex: 10,
        }}
      >
        <div style={{ display: "flex", alignItems: "center", gap: "10px" }}>
          <Image
            src="/horizonlogo.png"
            alt="Horizon Labs Logo"
            width={100}
            height={32}
            style={{ objectFit: "contain", opacity: 0.8 }}
          />
        </div>
        <p>Built with ⚡ by Horizon Labs • Hacker House Goa 2026 • Vagator Node [15.60° N, 73.74° E]</p>
      </footer>
    </div>
  );
}
