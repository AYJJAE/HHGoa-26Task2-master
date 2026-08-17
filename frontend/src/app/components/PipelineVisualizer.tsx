"use client";

import React from "react";
import { PipelineStage } from "./VoiceRecorder";

interface PipelineVisualizerProps {
  currentStage: PipelineStage;
  latencies?: {
    stt_ms?: number;
    retrieval_ms?: number;
    generation_ms?: number;
    total_ms?: number;
  };
}

export default function PipelineVisualizer({ currentStage, latencies }: PipelineVisualizerProps) {
  const stages = [
    { key: "VOICE", label: "Voice", icon: "🎙️", latency: undefined },
    { key: "STT", label: "STT", icon: "⚡", latency: latencies?.stt_ms },
    { key: "RETRIEVAL", label: "Hybrid Search", icon: "🔍", latency: latencies?.retrieval_ms },
    { key: "RAG", label: "Generation", icon: "🧠", latency: latencies?.generation_ms },
    { key: "ANSWER", label: "Answer", icon: "✨", latency: undefined },
  ];

  const getStageStatus = (stageKey: string) => {
    switch (currentStage) {
      case "IDLE":
        return "idle";
      case "LISTENING":
        return stageKey === "VOICE" ? "active" : "idle";
      case "PROCESSING":
      case "TRANSCRIBING":
        if (stageKey === "VOICE") return "completed";
        if (stageKey === "STT") return "active";
        return "idle";
      case "RETRIEVING":
        if (["VOICE", "STT"].includes(stageKey)) return "completed";
        if (stageKey === "RETRIEVAL") return "active";
        return "idle";
      case "GENERATING":
        if (["VOICE", "STT", "RETRIEVAL"].includes(stageKey)) return "completed";
        if (stageKey === "RAG") return "active";
        return "idle";
      case "ANSWER":
        return "completed";
      case "ERROR":
        return "idle";
      default:
        return "idle";
    }
  };

  return (
    <div
      style={{
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        gap: "8px",
        flexWrap: "wrap",
        padding: "10px 16px",
        background: "rgba(17, 23, 34, 0.75)",
        backdropFilter: "blur(14px)",
        borderRadius: "14px",
        border: "1px solid var(--border-subtle)",
        maxWidth: "760px",
        width: "100%",
        margin: "0 auto",
      }}
    >
      {stages.map((st, idx) => {
        const status = getStageStatus(st.key);
        const isActive = status === "active";
        const isCompleted = status === "completed";

        return (
          <React.Fragment key={st.key}>
            <div
              style={{
                display: "flex",
                alignItems: "center",
                gap: "7px",
                padding: "6px 12px",
                borderRadius: "10px",
                background: isActive
                  ? "rgba(0, 168, 120, 0.16)"
                  : isCompleted
                  ? "rgba(22, 199, 132, 0.10)"
                  : "rgba(255, 255, 255, 0.02)",
                border: isActive
                  ? "1px solid rgba(0, 168, 120, 0.55)"
                  : isCompleted
                  ? "1px solid rgba(22, 199, 132, 0.35)"
                  : "1px solid rgba(255, 255, 255, 0.04)",
                boxShadow: isActive ? "0 0 14px rgba(0, 168, 120, 0.25)" : "none",
                transition: "all 0.2s ease",
              }}
            >
              <span style={{ fontSize: "0.85rem" }}>
                {isCompleted ? "✓" : st.icon}
              </span>
              <span
                style={{
                  fontSize: "0.75rem",
                  fontWeight: 700,
                  color: isActive
                    ? "var(--tropical-green)"
                    : isCompleted
                    ? "#34D399"
                    : "var(--text-muted)",
                  textTransform: "uppercase",
                  letterSpacing: "0.04em",
                }}
              >
                {st.label}
              </span>

              {st.latency !== undefined && st.latency > 0 && isCompleted && (
                <span
                  style={{
                    fontFamily: "var(--font-mono)",
                    fontSize: "0.68rem",
                    fontWeight: 600,
                    color: "var(--text-primary)",
                    background: "rgba(0,0,0,0.4)",
                    padding: "2px 6px",
                    borderRadius: "4px",
                    border: "1px solid rgba(255,255,255,0.06)",
                  }}
                >
                  {Math.round(st.latency)}ms
                </span>
              )}
            </div>

            {idx < stages.length - 1 && (
              <span
                style={{
                  color: isCompleted ? "rgba(22, 199, 132, 0.55)" : "rgba(255, 255, 255, 0.12)",
                  fontSize: "0.8rem",
                  userSelect: "none",
                }}
              >
                →
              </span>
            )}
          </React.Fragment>
        );
      })}
    </div>
  );
}
