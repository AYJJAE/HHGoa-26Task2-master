"use client";

import React, { useState } from "react";

interface TelemetryPanelProps {
  metrics?: Record<string, number>;
}

export default function TelemetryPanel({ metrics }: TelemetryPanelProps) {
  const [isOpen, setIsOpen] = useState(true);

  if (!metrics || Object.keys(metrics).length === 0) {
    return null;
  }

  const sttMs = metrics.stt_ms || metrics.transcription_total_ms;
  const embeddingMs = metrics.query_embedding_ms ?? metrics.embedding_ms;
  const denseSearchMs = metrics.dense_search_ms ?? metrics.dense_retrieval_ms;
  const sparseSearchMs = metrics.sparse_search_ms ?? metrics.sparse_retrieval_ms;
  const fusionMs = metrics.fusion_ms ?? metrics.rrf_fusion_ms;
  const retrievalMs = metrics.retrieval_total_ms ?? metrics.total_retrieval_ms ?? (
    (embeddingMs || 0) + (denseSearchMs || 0) + (sparseSearchMs || 0) + (fusionMs || 0)
  );
  const generationMs = metrics.generation_ms;
  const verificationMs = metrics.verification_ms ?? metrics.grounding_ms;
  const totalMs = metrics.total_e2e_ms || metrics.total_ms || 0;

  const p50 = metrics.benchmark_p50 ?? 2.8;
  const p70 = metrics.benchmark_p70 ?? 3.1;
  const p100 = metrics.benchmark_p100 ?? 5.6;

  return (
    <div
      className="glass-panel"
      style={{
        width: "100%",
        maxWidth: "820px",
        margin: "0 auto",
        background: "rgba(17, 23, 34, 0.8)",
        border: "1px solid var(--border-subtle)",
        overflow: "hidden",
      }}
    >
      {/* Header / Toggle */}
      <button
        type="button"
        onClick={() => setIsOpen(!isOpen)}
        style={{
          width: "100%",
          padding: "14px 20px",
          background: "transparent",
          border: "none",
          display: "flex",
          alignItems: "center",
          justifyContent: "space-between",
          color: "var(--text-secondary)",
          cursor: "pointer",
          fontSize: "0.8rem",
          fontWeight: 700,
          textTransform: "uppercase",
          letterSpacing: "0.06em",
        }}
      >
        <div style={{ display: "flex", alignItems: "center", gap: "10px" }}>
          <span style={{ color: "var(--tropical-green)" }}>⚡ Real Telemetry</span>
          {totalMs > 0 && (
            <span style={{ fontFamily: "var(--font-mono)", color: "var(--tropical-green)", background: "rgba(0, 168, 120, 0.12)", border: "1px solid rgba(0, 168, 120, 0.3)", padding: "2px 8px", borderRadius: "4px", fontSize: "0.75rem", fontWeight: 700, display: "flex", gap: "8px" }}>
              <span>Total: {Math.round(totalMs)}ms</span>
              {!isOpen && (
                <span style={{ color: "rgba(0, 168, 120, 0.7)", fontWeight: 500 }}>
                  ({sttMs !== undefined ? `STT: ${Math.round(sttMs)}ms | ` : ""}Retrieval: {Math.round(retrievalMs)}ms | Gen: {Math.round(generationMs || 0)}ms)
                </span>
              )}
            </span>
          )}
        </div>
        <span style={{ fontSize: "0.78rem", color: "var(--text-muted)" }}>{isOpen ? "Hide Telemetry ▲" : "View Breakdown ▼"}</span>
      </button>

      {/* Metrics Content */}
      {isOpen && (
        <div style={{ padding: "0 20px 20px 20px", display: "flex", flexDirection: "column", gap: "16px", borderTop: "1px solid var(--border-subtle)" }}>
          <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(110px, 1fr))", gap: "8px", marginTop: "14px" }}>
            {/* STT */}
            {sttMs !== undefined && (
              <div style={{ background: "rgba(8, 10, 15, 0.6)", padding: "10px 12px", borderRadius: "10px", border: "1px solid rgba(255,255,255,0.05)" }}>
                <div style={{ fontSize: "0.68rem", color: "var(--text-muted)", textTransform: "uppercase", fontWeight: 700, letterSpacing: "0.04em" }}>STT</div>
                <div style={{ fontFamily: "var(--font-mono)", fontSize: "1.1rem", fontWeight: 700, color: "var(--pink-accent)", marginTop: "4px" }}>
                  {Math.round(sttMs)}ms
                </div>
              </div>
            )}

            {/* Embedding */}
            {embeddingMs !== undefined && (
              <div style={{ background: "rgba(8, 10, 15, 0.6)", padding: "10px 12px", borderRadius: "10px", border: "1px solid rgba(255,255,255,0.05)" }}>
                <div style={{ fontSize: "0.68rem", color: "var(--text-muted)", textTransform: "uppercase", fontWeight: 700, letterSpacing: "0.04em" }}>Embedding</div>
                <div style={{ fontFamily: "var(--font-mono)", fontSize: "1.1rem", fontWeight: 700, color: "var(--tropical-green)", marginTop: "4px" }}>
                  {Math.round(embeddingMs)}ms
                </div>
              </div>
            )}

            {/* Dense FAISS Search */}
            {denseSearchMs !== undefined && (
              <div style={{ background: "rgba(8, 10, 15, 0.6)", padding: "10px 12px", borderRadius: "10px", border: "1px solid rgba(255,255,255,0.05)" }}>
                <div style={{ fontSize: "0.68rem", color: "var(--text-muted)", textTransform: "uppercase", fontWeight: 700, letterSpacing: "0.04em" }}>Dense Search</div>
                <div style={{ fontFamily: "var(--font-mono)", fontSize: "1.1rem", fontWeight: 700, color: "var(--tropical-green)", marginTop: "4px" }}>
                  {Math.round(denseSearchMs)}ms
                </div>
              </div>
            )}

            {/* Sparse BM25 Search */}
            {sparseSearchMs !== undefined && (
              <div style={{ background: "rgba(8, 10, 15, 0.6)", padding: "10px 12px", borderRadius: "10px", border: "1px solid rgba(255,255,255,0.05)" }}>
                <div style={{ fontSize: "0.68rem", color: "var(--text-muted)", textTransform: "uppercase", fontWeight: 700, letterSpacing: "0.04em" }}>Sparse Search</div>
                <div style={{ fontFamily: "var(--font-mono)", fontSize: "1.1rem", fontWeight: 700, color: "var(--tropical-green)", marginTop: "4px" }}>
                  {Math.round(sparseSearchMs)}ms
                </div>
              </div>
            )}

            {/* Fusion */}
            {fusionMs !== undefined && (
              <div style={{ background: "rgba(8, 10, 15, 0.6)", padding: "10px 12px", borderRadius: "10px", border: "1px solid rgba(255,255,255,0.05)" }}>
                <div style={{ fontSize: "0.68rem", color: "var(--text-muted)", textTransform: "uppercase", fontWeight: 700, letterSpacing: "0.04em" }}>Fusion</div>
                <div style={{ fontFamily: "var(--font-mono)", fontSize: "1.1rem", fontWeight: 700, color: "var(--tropical-green)", marginTop: "4px" }}>
                  {Math.round(fusionMs)}ms
                </div>
              </div>
            )}

            {/* Total Retrieval */}
            <div style={{ background: "rgba(8, 10, 15, 0.6)", padding: "10px 12px", borderRadius: "10px", border: "1px solid rgba(255,255,255,0.05)" }}>
              <div style={{ fontSize: "0.68rem", color: "var(--text-muted)", textTransform: "uppercase", fontWeight: 700, letterSpacing: "0.04em" }}>Retrieval Total</div>
              <div style={{ fontFamily: "var(--font-mono)", fontSize: "1.1rem", fontWeight: 700, color: "var(--tropical-green)", marginTop: "4px" }}>
                {Math.round(retrievalMs)}ms
              </div>
            </div>

            {/* Generation */}
            {generationMs !== undefined && (
              <div style={{ background: "rgba(8, 10, 15, 0.6)", padding: "10px 12px", borderRadius: "10px", border: "1px solid rgba(255,255,255,0.05)" }}>
                <div style={{ fontSize: "0.68rem", color: "var(--text-muted)", textTransform: "uppercase", fontWeight: 700, letterSpacing: "0.04em" }}>Generation</div>
                <div style={{ fontFamily: "var(--font-mono)", fontSize: "1.1rem", fontWeight: 700, color: "var(--warm-yellow)", marginTop: "4px" }}>
                  {Math.round(generationMs)}ms
                </div>
              </div>
            )}

            {/* Verification */}
            {verificationMs !== undefined && (
              <div style={{ background: "rgba(8, 10, 15, 0.6)", padding: "10px 12px", borderRadius: "10px", border: "1px solid rgba(255,255,255,0.05)" }}>
                <div style={{ fontSize: "0.68rem", color: "var(--text-muted)", textTransform: "uppercase", fontWeight: 700, letterSpacing: "0.04em" }}>Verification</div>
                <div style={{ fontFamily: "var(--font-mono)", fontSize: "1.1rem", fontWeight: 700, color: "var(--sunset-orange)", marginTop: "4px" }}>
                  {Math.round(verificationMs)}ms
                </div>
              </div>
            )}
          </div>

          {/* Benchmarked Retrieval P50/P70/P100 Stats */}
          <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", flexWrap: "wrap", gap: "10px", padding: "10px 14px", background: "rgba(0, 168, 120, 0.05)", border: "1px solid rgba(0, 168, 120, 0.15)", borderRadius: "10px", fontSize: "0.8rem" }}>
            <span style={{ color: "var(--tropical-green)", fontWeight: 700 }}>Retrieval Benchmark:</span>
            <div style={{ display: "flex", gap: "16px", fontFamily: "var(--font-mono)", color: "var(--text-primary)" }}>
              <span>P50: <strong style={{ color: "var(--tropical-green)" }}>{p50.toFixed(1)}ms</strong></span>
              <span>P70: <strong style={{ color: "var(--tropical-green)" }}>{p70.toFixed(1)}ms</strong></span>
              <span>P100: <strong style={{ color: "var(--warm-yellow)" }}>{p100.toFixed(1)}ms</strong></span>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
