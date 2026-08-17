"use client";

import React, { useEffect, useState } from "react";

export type PipelineStage = 
  | "IDLE" 
  | "LISTENING" 
  | "PROCESSING" 
  | "TRANSCRIBING" 
  | "RETRIEVING" 
  | "GENERATING" 
  | "ANSWER" 
  | "ERROR";

interface VoiceRecorderProps {
  stage: PipelineStage;
  isRecording: boolean;
  onStartRecording: () => void;
  onStopRecording: () => void;
  onCancelRecording?: () => void;
  errorMessage?: string | null;
}

export default function VoiceRecorder({
  stage,
  isRecording,
  onStartRecording,
  onStopRecording,
  onCancelRecording,
  errorMessage,
}: VoiceRecorderProps) {
  const [seconds, setSeconds] = useState(0);

  useEffect(() => {
    if (!isRecording) {
      return;
    }
    const interval = setInterval(() => {
      setSeconds((prev) => prev + 1);
    }, 1000);
    return () => {
      clearInterval(interval);
      setSeconds(0);
    };
  }, [isRecording]);

  const formatTime = (secs: number) => {
    const m = Math.floor(secs / 60).toString().padStart(2, "0");
    const s = (secs % 60).toString().padStart(2, "0");
    return `${m}:${s}`;
  };

  const getStageConfig = () => {
    switch (stage) {
      case "LISTENING":
        return {
          title: "Listening to voice input...",
          color: "var(--pink-accent)",
          sub: "Speak in English, हिन्दी, मराठी, or कोंकणी",
        };
      case "PROCESSING":
        return {
          title: "Packaging audio buffer...",
          color: "var(--sunset-orange)",
          sub: "Formatting for speech recognition",
        };
      case "TRANSCRIBING":
        return {
          title: "Transcribing with STT (ElevenLabs / Sarvam)...",
          color: "var(--sunset-orange)",
          sub: "Converting spoken speech to text",
        };
      case "RETRIEVING":
        return {
          title: "Searching hybrid index (BGE-M3 + BM25)...",
          color: "var(--ocean-blue)",
          sub: "Querying dense vector & sparse keyword index",
        };
      case "GENERATING":
        return {
          title: "Synthesizing grounded answer...",
          color: "var(--warm-yellow)",
          sub: "Verifying facts against retrieved context",
        };
      case "ANSWER":
        return {
          title: "Answer ready",
          color: "var(--tropical-green)",
          sub: "Tap to ask another question",
        };
      case "ERROR":
        return {
          title: errorMessage || "An error occurred",
          color: "var(--pink-accent)",
          sub: "Tap the microphone to try again",
        };
      case "IDLE":
      default:
        return {
          title: "Tap to speak",
          color: "var(--goa-green)",
          sub: "English • हिन्दी • मराठी • कोंकणी",
        };
    }
  };

  const isBusy = ["PROCESSING", "TRANSCRIBING", "RETRIEVING", "GENERATING"].includes(stage);
  const stageCfg = getStageConfig();

  return (
    <div style={{ display: "flex", flexDirection: "column", alignItems: "center", gap: "22px", padding: "12px 0" }}>
      
      {/* Centerpiece Microphone Interaction Container */}
      <div style={{ position: "relative", display: "flex", alignItems: "center", justifyContent: "center", minHeight: "130px" }}>
        
        {/* Active Radar Ripple Rings for Listening */}
        {isRecording && (
          <>
            <div
              style={{
                position: "absolute",
                width: "150px",
                height: "150px",
                borderRadius: "50%",
                background: "radial-gradient(circle, rgba(255, 79, 129, 0.25) 0%, rgba(0, 168, 120, 0.08) 60%, transparent 80%)",
                animation: "micRadarPulse 1.8s infinite ease-out",
                pointerEvents: "none",
              }}
            />
            <div
              style={{
                position: "absolute",
                width: "180px",
                height: "180px",
                borderRadius: "50%",
                border: "1px dashed rgba(255, 79, 129, 0.35)",
                animation: "rotateGradient 8s linear infinite",
                pointerEvents: "none",
              }}
            />
          </>
        )}

        {/* Ambient Subtle Radar Ring for Idle */}
        {!isRecording && !isBusy && (
          <div
            style={{
              position: "absolute",
              width: "128px",
              height: "128px",
              borderRadius: "50%",
              border: "1px solid rgba(0, 168, 120, 0.15)",
              pointerEvents: "none",
            }}
          />
        )}

        {/* Rotating Dual-Gradient Ring for Processing / Transcribing / Retrieving */}
        {isBusy && (
          <div
            style={{
              position: "absolute",
              width: "124px",
              height: "124px",
              borderRadius: "50%",
              padding: "2px",
              background: "conic-gradient(from 0deg, #FF7A3D, #00A878, #168AAD, #FFC857, #FF7A3D)",
              animation: "rotateGradient 1.5s linear infinite",
              WebkitMask: "radial-gradient(farthest-side, transparent calc(100% - 3px), #fff calc(100% - 2px))",
              mask: "radial-gradient(farthest-side, transparent calc(100% - 3px), #fff calc(100% - 2px))",
            }}
          />
        )}

        {/* Microphone Action Button */}
        <button
          type="button"
          onClick={() => {
            if (isRecording) {
              onStopRecording();
            } else if (!isBusy) {
              onStartRecording();
            }
          }}
          disabled={isBusy}
          aria-label={isRecording ? "Stop voice recording" : "Start voice recording"}
          style={{
            position: "relative",
            width: "102px",
            height: "102px",
            borderRadius: "50%",
            background: isRecording
              ? "linear-gradient(135deg, #FF4F81 0%, #E11D48 100%)"
              : isBusy
              ? "#111722"
              : "linear-gradient(135deg, #111722 0%, #151C28 100%)",
            border: isRecording
              ? "2px solid rgba(255, 255, 255, 0.6)"
              : isBusy
              ? "1px solid rgba(255, 255, 255, 0.1)"
              : "2px solid rgba(0, 168, 120, 0.45)",
            boxShadow: isRecording
              ? "0 0 36px rgba(255, 79, 129, 0.6), inset 0 0 16px rgba(255,255,255,0.2)"
              : isBusy
              ? "none"
              : "0 8px 32px rgba(0, 168, 120, 0.22), inset 0 1px 0 rgba(255, 255, 255, 0.1)",
            cursor: isBusy ? "not-allowed" : "pointer",
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            transition: "all 0.25s cubic-bezier(0.4, 0, 0.2, 1)",
            transform: isRecording ? "scale(1.06)" : "scale(1)",
            zIndex: 2,
          }}
        >
          {isRecording ? (
            /* Square Stop Icon */
            <div
              style={{
                width: "24px",
                height: "24px",
                background: "#FFFFFF",
                borderRadius: "5px",
                boxShadow: "0 2px 8px rgba(0,0,0,0.3)",
              }}
            />
          ) : isBusy ? (
            /* Rotating AI Compass / Radar Icon */
            <svg width="34" height="34" viewBox="0 0 24 24" fill="none" stroke="var(--sunset-orange)" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" style={{ animation: "rotateGradient 2s linear infinite" }}>
              <path d="M12 2v4M12 18v4M4.93 4.93l2.83 2.83M16.24 16.24l2.83 2.83M2 12h4M18 12h4M4.93 19.07l2.83-2.83M16.24 7.76l2.83-2.83" />
            </svg>
          ) : (
            /* Sleek Microphone Icon with Goa Green Accent */
            <svg
              width="38"
              height="38"
              viewBox="0 0 24 24"
              fill="none"
              stroke="#16C784"
              strokeWidth="2"
              strokeLinecap="round"
              strokeLinejoin="round"
              style={{ filter: "drop-shadow(0 2px 6px rgba(0, 168, 120, 0.4))" }}
            >
              <path d="M12 2a3 3 0 0 0-3 3v7a3 3 0 0 0 6 0V5a3 3 0 0 0-3-3Z" />
              <path d="M19 10v2a7 7 0 0 1-14 0v-2" />
              <line x1="12" x2="12" y1="19" y2="22" />
            </svg>
          )}
        </button>
      </div>

      {/* Recording Waveform & Live Timer Bar */}
      {isRecording && (
        <div
          style={{
            display: "flex",
            alignItems: "center",
            gap: "14px",
            background: "rgba(17, 23, 34, 0.9)",
            padding: "8px 20px",
            borderRadius: "9999px",
            border: "1px solid rgba(255, 79, 129, 0.4)",
            boxShadow: "0 4px 20px rgba(0,0,0,0.5)",
          }}
        >
          {/* Pulsing Recording Dot */}
          <div
            style={{
              width: "8px",
              height: "8px",
              borderRadius: "50%",
              background: "var(--pink-accent)",
              boxShadow: "0 0 10px var(--pink-accent)",
              animation: "pulseDot 1s infinite alternate",
            }}
          />

          {/* Time Counter */}
          <span style={{ fontFamily: "var(--font-mono)", fontSize: "0.95rem", fontWeight: 700, color: "#F5F7FA" }}>
            {formatTime(seconds)}
          </span>

          {/* Dynamic Frequency Bars */}
          <div style={{ display: "flex", alignItems: "center", gap: "3px", height: "18px" }}>
            {[10, 18, 14, 22, 12, 16, 20, 14, 18, 10].map((h, i) => (
              <div
                key={i}
                style={{
                  width: "3px",
                  height: `${h}px`,
                  background: i % 2 === 0 ? "var(--pink-accent)" : "var(--tropical-green)",
                  borderRadius: "2px",
                  animation: "listeningWaveAnim 0.55s infinite alternate ease-in-out",
                  animationDelay: `${i * 0.06}s`,
                }}
              />
            ))}
          </div>

          {/* Cancel Button */}
          {onCancelRecording && (
            <button
              type="button"
              onClick={onCancelRecording}
              style={{
                background: "transparent",
                border: "none",
                color: "var(--text-muted)",
                fontSize: "0.8rem",
                fontWeight: 600,
                cursor: "pointer",
                padding: "2px 6px",
                transition: "color 0.2s ease",
              }}
              onMouseEnter={(e) => (e.currentTarget.style.color = "var(--pink-accent)")}
              onMouseLeave={(e) => (e.currentTarget.style.color = "var(--text-muted)")}
            >
              Cancel
            </button>
          )}
        </div>
      )}

      {/* Stage Status and Subtitle */}
      <div style={{ textAlign: "center", display: "flex", flexDirection: "column", gap: "4px" }}>
        <p style={{ fontSize: "1.05rem", fontWeight: 700, color: stageCfg.color, letterSpacing: "-0.01em" }}>
          {stageCfg.title}
        </p>
        <p style={{ fontSize: "0.85rem", color: "var(--text-secondary)" }}>
          {stageCfg.sub}
        </p>
      </div>
    </div>
  );
}
