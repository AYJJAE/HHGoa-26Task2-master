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

/**
 * Safely resolves the API Base URL from Vite or Next.js environment variables.
 * Trailing slashes are stripped and https:// protocol is enforced for remote hosts.
 */
function getApiBaseUrl(): string {
  // @ts-ignore
  const viteEnv = typeof import.meta !== "undefined" && (import.meta as any).env?.VITE_API_BASE_URL;
  const nextEnv = process.env.VITE_API_BASE_URL || process.env.NEXT_PUBLIC_API_URL;
  let raw = String(viteEnv || nextEnv || "").trim().replace(/\/+$/, "");

  if (raw && !raw.startsWith("http://") && !raw.startsWith("https://") && !raw.startsWith("/")) {
    const protocol = raw.includes("localhost") || raw.includes("127.0.0.1") ? "http://" : "https://";
    raw = `${protocol}${raw}`;
  }

  return raw;
}

interface VoiceMetadata {
  transcriptText?: string;
  sttProvider?: string;
  sttLanguage?: string;
  sttFallbackUsed?: boolean;
  sttLatencyMs?: number;
}

interface ConversationTurn {
  id: string;
  query: string;
  answer: string;
  timestamp: string;
  grounded?: boolean;
  confidence?: string;
  category?: string;
}

type GoaCategory =
  | "ALL"
  | "BEACHES"
  | "ITINERARIES"
  | "FOOD"
  | "HERITAGE"
  | "CULTURE"
  | "NATURE"
  | "SUNSET"
  | "TRANSPORT";

const CATEGORY_TABS: { id: GoaCategory; label: string; icon: string }[] = [
  { id: "ALL", label: "All Topics", icon: "✨" },
  { id: "BEACHES", label: "Beaches", icon: "🏖️" },
  { id: "ITINERARIES", label: "Itineraries", icon: "🗺️" },
  { id: "FOOD", label: "Goan Food", icon: "🍲" },
  { id: "HERITAGE", label: "Old Goa & Heritage", icon: "🏛️" },
  { id: "SUNSET", label: "Sunset Spots", icon: "🌅" },
  { id: "CULTURE", label: "Culture & Festivals", icon: "🥥" },
  { id: "NATURE", label: "Nature & Family", icon: "🌴" },
  { id: "TRANSPORT", label: "Transport & Travel", icon: "🛵" },
];

const GOA_PROMPTS: { label: string; lang: string; category: GoaCategory }[] = [
  { label: "Best beaches in Goa?", lang: "EN", category: "BEACHES" },
  { label: "What are the main beaches in North Goa?", lang: "EN", category: "BEACHES" },
  { label: "Plan a 2 day Goa trip.", lang: "EN", category: "ITINERARIES" },
  { label: "Give me a relaxed itinerary for Goa.", lang: "EN", category: "ITINERARIES" },
  { label: "Plan my Goa trip for 3 days.", lang: "EN", category: "ITINERARIES" },
  { label: "What Goan food should I try?", lang: "EN", category: "FOOD" },
  { label: "What should I visit in Old Goa?", lang: "EN", category: "HERITAGE" },
  { label: "Best places for sunset in Goa?", lang: "EN", category: "SUNSET" },
  { label: "Tell me about Goa's culture.", lang: "EN", category: "CULTURE" },
  { label: "Which places are good for families in Goa?", lang: "EN", category: "NATURE" },
  { label: "What can I do near Panjim?", lang: "EN", category: "HERITAGE" },
  { label: "Best places to explore in South Goa?", lang: "EN", category: "BEACHES" },
  { label: "How to get around and transport in Goa?", lang: "EN", category: "TRANSPORT" },
  { label: "गोवा में 2 दिन का ट्रिप कैसे प्लान करें?", lang: "HI", category: "ITINERARIES" },
  { label: "गोव्यात दोन दिवसांचा प्लॅन कसा करावा?", lang: "MR", category: "ITINERARIES" },
  { label: "गोवा राज्याची राजधानी कोणती?", lang: "KOK", category: "HERITAGE" },
];

export default function GoaAssistantPage() {
  const [stage, setStage] = useState<PipelineStage>("IDLE");
  const [isRecording, setIsRecording] = useState(false);
  const [answerData, setAnswerData] = useState<AnswerData | null>(null);
  const [latencyMetrics, setLatencyMetrics] = useState<Record<string, number> | undefined>();
  const [textInput, setTextInput] = useState("");
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [activeCategory, setActiveCategory] = useState<GoaCategory>("ALL");
  const [conversationHistory, setConversationHistory] = useState<ConversationTurn[]>([]);
  const [showHistoryDrawer, setShowHistoryDrawer] = useState(false);

  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const chunksRef = useRef<BlobPart[]>([]);
  const currentRequestIdRef = useRef<number>(0);
  const requestIdCounter = useRef<number>(0);

  // Helper to format conversation history into [{role, content}]
  const getFormattedHistory = () => {
    return conversationHistory.map((turn) => [
      { role: "user", content: turn.query },
      { role: "assistant", content: turn.answer },
    ]).flat();
  };

  // --- Voice Recording Lifecycle ---
  const startRecording = async () => {
    setErrorMessage(null);
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });

      let mimeType = "audio/webm";
      if (typeof MediaRecorder !== "undefined" && !MediaRecorder.isTypeSupported(mimeType)) {
        if (MediaRecorder.isTypeSupported("audio/mp4")) {
          mimeType = "audio/mp4";
        } else if (MediaRecorder.isTypeSupported("audio/ogg")) {
          mimeType = "audio/ogg";
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

    const historyPayload = getFormattedHistory();
    if (historyPayload.length > 0) {
      formData.append("history", JSON.stringify(historyPayload));
    }

    const apiBase = getApiBaseUrl();
    const requestUrl = `${apiBase}/api/voice_ask`;
    console.log(`[Goa Assistant] Voice Ask URL: ${requestUrl}`);

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

      const response = await fetch(requestUrl, {
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

      const newAnswerData: AnswerData = {
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
      };

      setAnswerData(newAnswerData);

      // Append to conversation history
      if (newAnswerData.answer) {
        setConversationHistory((prev) => [
          ...prev,
          {
            id: String(Date.now()),
            query: transcript,
            answer: newAnswerData.answer || "",
            timestamp: new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }),
            grounded: newAnswerData.grounded,
            confidence: newAnswerData.retrievalConfidence,
            category: data.category,
          },
        ]);
      }

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

    const apiBase = getApiBaseUrl();
    const requestUrl = `${apiBase}/api/ask`;
    console.log(`[Goa Assistant] Text Ask URL: ${requestUrl}`);

    try {
      setTimeout(() => {
        if (currentRequestIdRef.current === requestId && stage !== "ERROR") {
          setStage("GENERATING");
        }
      }, 400);

      const historyPayload = getFormattedHistory();

      const response = await fetch(requestUrl, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          query: queryText,
          history: historyPayload.length > 0 ? historyPayload : undefined,
        }),
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

      const newAnswerData: AnswerData = {
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
      };

      setAnswerData(newAnswerData);

      // Append to conversation history
      if (newAnswerData.answer) {
        setConversationHistory((prev) => [
          ...prev,
          {
            id: String(Date.now()),
            query: queryText,
            answer: newAnswerData.answer || "",
            timestamp: new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }),
            grounded: newAnswerData.grounded,
            confidence: newAnswerData.retrievalConfidence,
            category: ragResponse.category,
          },
        ]);
      }

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

  const filteredPrompts = activeCategory === "ALL"
    ? GOA_PROMPTS
    : GOA_PROMPTS.filter((p) => p.category === activeCategory);

  return (
    <div style={{ position: "relative", minHeight: "100vh", display: "flex", flexDirection: "column", backgroundColor: "var(--bg-primary)" }}>
      {/* 1. Goa Digital Map Background Layer */}
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
                  background: "linear-gradient(135deg, rgba(255, 79, 129, 0.25) 0%, rgba(0, 168, 120, 0.2) 100%)",
                  border: "1px solid rgba(255, 79, 129, 0.4)",
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "center",
                  fontSize: "1.1rem",
                }}
              >
                🌴
              </div>

              <div style={{ display: "flex", flexDirection: "column" }}>
                <div style={{ display: "flex", alignItems: "center", gap: "8px" }}>
                  <span className="font-display" style={{ fontSize: "0.95rem", fontWeight: 800, color: "#F5F7FA" }}>
                    Goa Companion
                  </span>
                  <span className="badge badge-node" style={{ fontSize: "0.62rem", padding: "1px 6px" }}>
                    🌴 VAGATOR NODE • 15.60° N
                  </span>
                </div>
              </div>
            </div>
          </div>

          {/* Navigation links & Languages */}
          <div style={{ display: "flex", alignItems: "center", gap: "10px", flexWrap: "wrap" }}>
            <div className="badge badge-lang" style={{ gap: "8px", padding: "5px 12px" }}>
              <span style={{ color: "var(--tropical-green)", fontWeight: 700 }}>EN</span> • <span>हिन्दी</span> • <span>मराठी</span> • <span>कोंकणी</span>
            </div>

            <Link
              href="/rag"
              className="btn-secondary"
              style={{
                padding: "6px 12px",
                fontSize: "0.78rem",
                textDecoration: "none",
              }}
            >
              Voice RAG 🎙️
            </Link>

            <span className="badge badge-live">
              <span style={{ width: "6px", height: "6px", borderRadius: "50%", background: "#16C784", animation: "pulseDot 1.5s infinite" }} />
              Live Companion
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
          maxWidth: "920px",
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
                color: "var(--pink-accent)",
                background: "rgba(255, 79, 129, 0.12)",
                border: "1px solid rgba(255, 79, 129, 0.35)",
                padding: "4px 14px",
                borderRadius: "9999px",
                display: "inline-flex",
                alignItems: "center",
                gap: "6px",
              }}
            >
              <span>🌴</span> Goa AI Island Companion • Multilingual RAG
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
            Your AI Companion for <span className="gradient-goa-accent">Goa</span>
          </h1>

          <p style={{ fontSize: "1.05rem", color: "var(--text-secondary)", maxWidth: "620px", margin: "0 auto", lineHeight: 1.6 }}>
            Ask anything about beaches, customized 1–5 day itineraries, local Goan cuisine, Old Goa heritage, sunsets, and transport in English, Hindi, Marathi, or Konkani.
          </p>
        </section>

        {/* Goa Category Filter Chips */}
        <section
          style={{
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            flexWrap: "wrap",
            gap: "8px",
          }}
        >
          {CATEGORY_TABS.map((tab) => {
            const isSelected = activeCategory === tab.id;
            return (
              <button
                key={tab.id}
                type="button"
                onClick={() => setActiveCategory(tab.id)}
                style={{
                  background: isSelected
                    ? "linear-gradient(135deg, rgba(0, 168, 120, 0.25) 0%, rgba(22, 138, 173, 0.25) 100%)"
                    : "rgba(255, 255, 255, 0.03)",
                  border: isSelected
                    ? "1px solid rgba(0, 168, 120, 0.6)"
                    : "1px solid rgba(255, 255, 255, 0.08)",
                  color: isSelected ? "var(--tropical-green)" : "var(--text-secondary)",
                  padding: "6px 14px",
                  borderRadius: "9999px",
                  fontSize: "0.82rem",
                  fontWeight: isSelected ? 700 : 500,
                  cursor: "pointer",
                  display: "flex",
                  alignItems: "center",
                  gap: "6px",
                  transition: "all 0.2s ease",
                  boxShadow: isSelected ? "0 0 14px rgba(0, 168, 120, 0.2)" : "none",
                }}
                onMouseEnter={(e) => {
                  if (!isSelected) {
                    e.currentTarget.style.borderColor = "rgba(0, 168, 120, 0.35)";
                    e.currentTarget.style.color = "var(--text-primary)";
                  }
                }}
                onMouseLeave={(e) => {
                  if (!isSelected) {
                    e.currentTarget.style.borderColor = "rgba(255, 255, 255, 0.08)";
                    e.currentTarget.style.color = "var(--text-secondary)";
                  }
                }}
              >
                <span>{tab.icon}</span>
                <span>{tab.label}</span>
              </button>
            );
          })}
        </section>

        {/* Centerpiece Microphone Voice & Text Interaction Panel */}
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
              maxWidth: "640px",
              margin: "22px auto 0 auto",
              display: "flex",
              gap: "8px",
            }}
          >
            <input
              type="text"
              placeholder="Ask about Goa (e.g. Best beaches, 2 day trip, Goan food, Old Goa)..."
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

          {/* 1-Click Goa Suggestion Chips */}
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
            <span style={{ fontSize: "0.75rem", color: "var(--text-muted)", fontWeight: 600 }}>Explore:</span>
            {filteredPrompts.slice(0, 6).map((p, idx) => (
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
                  padding: "4px 11px",
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
            retrieval_ms:
              (latencyMetrics?.dense_retrieval_ms || 0) +
              (latencyMetrics?.sparse_retrieval_ms || 0) +
              (latencyMetrics?.rrf_fusion_ms || 0),
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

        {/* Multi-Turn Conversation Log */}
        {conversationHistory.length > 1 && (
          <div
            className="glass-panel"
            style={{
              padding: "18px 22px",
              background: "rgba(13, 17, 24, 0.8)",
              border: "1px solid var(--border-subtle)",
              borderRadius: "16px",
            }}
          >
            <div
              style={{
                display: "flex",
                alignItems: "center",
                justifyContent: "space-between",
                marginBottom: showHistoryDrawer ? "14px" : "0",
                cursor: "pointer",
              }}
              onClick={() => setShowHistoryDrawer(!showHistoryDrawer)}
            >
              <div style={{ display: "flex", alignItems: "center", gap: "8px" }}>
                <span style={{ fontSize: "1rem" }}>💬</span>
                <span style={{ fontSize: "0.85rem", fontWeight: 700, color: "var(--text-primary)" }}>
                  Conversation Context ({conversationHistory.length} turns)
                </span>
                <span className="badge" style={{ fontSize: "0.68rem", background: "rgba(0, 168, 120, 0.15)", color: "var(--tropical-green)" }}>
                  Follow-ups active
                </span>
              </div>
              <span style={{ fontSize: "0.8rem", color: "var(--text-muted)" }}>
                {showHistoryDrawer ? "▲ Hide" : "▼ View previous turns"}
              </span>
            </div>

            {showHistoryDrawer && (
              <div style={{ display: "flex", flexDirection: "column", gap: "10px", marginTop: "12px" }}>
                {conversationHistory.map((turn, index) => (
                  <div
                    key={turn.id}
                    style={{
                      background: "rgba(21, 28, 40, 0.6)",
                      border: "1px solid rgba(255, 255, 255, 0.05)",
                      borderRadius: "10px",
                      padding: "10px 14px",
                      display: "flex",
                      flexDirection: "column",
                      gap: "4px",
                    }}
                  >
                    <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                      <span style={{ fontSize: "0.8rem", fontWeight: 700, color: "var(--tropical-green)" }}>
                        Q{index + 1}: {turn.query}
                      </span>
                      <span style={{ fontSize: "0.68rem", color: "var(--text-muted)" }}>{turn.timestamp}</span>
                    </div>
                    <p style={{ fontSize: "0.78rem", color: "var(--text-secondary)", margin: 0, lineHeight: 1.4 }}>
                      {turn.answer.length > 120 ? `${turn.answer.slice(0, 120)}...` : turn.answer}
                    </p>
                  </div>
                ))}
              </div>
            )}
          </div>
        )}

        {/* Performance & Real Telemetry Panel */}
        <TelemetryPanel metrics={latencyMetrics} />
      </main>

      {/* 4. Footer */}
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
        <p>© 2026 Hacker House Goa • Goa Companion • Built with ⚡ in Vagator [15.60° N, 73.74° E]</p>
      </footer>
    </div>
  );
}
