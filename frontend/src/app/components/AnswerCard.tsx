"use client";

import React, { useState } from "react";

export interface SourceChunk {
  document_id?: string;
  relevance?: number;
  strategy?: string;
  language?: string;
  text: string;
}

export interface AnswerData {
  query: string;
  transcriptText?: string;
  sttProvider?: string;
  sttLanguage?: string;
  sttFallbackUsed?: boolean;
  answer?: string;
  grounded?: boolean;
  contextSufficient?: boolean;
  refusalReason?: string;
  sources?: SourceChunk[];
  retrievalConfidence?: string;
}

interface AnswerCardProps {
  data: AnswerData;
  onEditQuery?: (editedQuery: string) => void;
}

export default function AnswerCard({ data, onEditQuery }: AnswerCardProps) {
  const [showContext, setShowContext] = useState(false);
  const [copied, setCopied] = useState(false);
  const [isEditingTranscript, setIsEditingTranscript] = useState(false);
  const [editedTranscript, setEditedTranscript] = useState(data.transcriptText || data.query);
  const [expandedSources, setExpandedSources] = useState<Record<number, boolean>>({});

  const handleCopy = () => {
    if (data.answer) {
      navigator.clipboard.writeText(data.answer);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    }
  };

  const toggleSourceExpand = (index: number) => {
    setExpandedSources((prev) => ({ ...prev, [index]: !prev[index] }));
  };

  const getLanguageLabel = (code?: string) => {
    if (!code) return "English";
    const map: Record<string, string> = {
      en: "English",
      "en-in": "English",
      hi: "Hindi",
      "hi-in": "Hindi",
      mr: "Marathi",
      "mr-in": "Marathi",
      kok: "Konkani",
      "kok-in": "Konkani",
      "code-mixed": "Code-mixed",
    };
    return map[code.toLowerCase()] || code.toUpperCase();
  };

  const confidence = (data.retrievalConfidence || "HIGH").toUpperCase();

  const getCardStyle = () => {
    if (!data.grounded || data.refusalReason) {
      return {
        borderColor: "rgba(255, 79, 129, 0.4)",
        glow: "0 0 30px rgba(255, 79, 129, 0.15)",
        badgeBg: "rgba(255, 79, 129, 0.15)",
        badgeColor: "var(--pink-accent)",
      };
    }
    if (confidence === "HIGH") {
      return {
        borderColor: "rgba(0, 168, 120, 0.4)",
        glow: "0 0 30px rgba(0, 168, 120, 0.15)",
        badgeBg: "rgba(0, 168, 120, 0.15)",
        badgeColor: "var(--tropical-green)",
      };
    }
    if (confidence === "MEDIUM") {
      return {
        borderColor: "rgba(255, 200, 87, 0.4)",
        glow: "0 0 30px rgba(255, 200, 87, 0.15)",
        badgeBg: "rgba(255, 200, 87, 0.15)",
        badgeColor: "var(--warm-yellow)",
      };
    }
    return {
      borderColor: "rgba(255, 122, 61, 0.4)",
      glow: "0 0 30px rgba(255, 122, 61, 0.15)",
      badgeBg: "rgba(255, 122, 61, 0.15)",
      badgeColor: "var(--sunset-orange)",
    };
  };

  const cardStyle = getCardStyle();

  const getGroundingText = () => {
    if (data.grounded && confidence === "HIGH") {
      return "✓ Answer verified and grounded in retrieved dataset context";
    }
    if (data.grounded && confidence === "MEDIUM") {
      return "⚠ Answer supported, but confidence is moderate";
    }
    if (data.grounded && confidence === "LOW") {
      return "⚠ Limited evidence in retrieved dataset";
    }
    return "✕ Insufficient relevant evidence in knowledge base";
  };

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: "20px", width: "100%", maxWidth: "820px", margin: "0 auto" }}>
      
      {/* 1. "YOU SAID" Card (Goa Night Map Node Card) */}
      <div
        className="glass-panel"
        style={{
          padding: "20px 24px",
          background: "rgba(17, 23, 34, 0.85)",
          border: "1px solid var(--border-subtle)",
          position: "relative",
          overflow: "hidden",
        }}
      >
        {/* Subtle Node Accent Stripe */}
        <div style={{ position: "absolute", top: 0, left: 0, width: "4px", height: "100%", background: "linear-gradient(to bottom, var(--goa-green), var(--ocean-blue))" }} />

        <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: "12px", flexWrap: "wrap", gap: "10px" }}>
          <div style={{ display: "flex", alignItems: "center", gap: "8px" }}>
            <span style={{ fontSize: "0.85rem" }}>📍</span>
            <span style={{ fontSize: "0.75rem", fontWeight: 800, letterSpacing: "0.08em", color: "var(--tropical-green)", textTransform: "uppercase" }}>
              YOU SAID
            </span>
          </div>

          {/* Provider & Language Metadata Chips */}
          <div style={{ display: "flex", alignItems: "center", gap: "8px", flexWrap: "wrap" }}>
            {data.sttProvider && (
              <span className={data.sttFallbackUsed ? "badge badge-provider-fallback" : "badge badge-provider-primary"}>
                {data.sttFallbackUsed ? "⚡ " : "✨ "}
                {data.sttProvider.toUpperCase()}
                {data.sttFallbackUsed ? " • FALLBACK" : " • PRIMARY"}
              </span>
            )}
            <span className="badge badge-lang" style={{ gap: "4px" }}>
              <span>🇮🇳</span> {getLanguageLabel(data.sttLanguage)}
            </span>
          </div>
        </div>

        {isEditingTranscript ? (
          <div style={{ display: "flex", gap: "10px", marginTop: "8px" }}>
            <input
              type="text"
              value={editedTranscript}
              onChange={(e) => setEditedTranscript(e.target.value)}
              style={{
                flex: 1,
                background: "rgba(8, 10, 15, 0.8)",
                border: "1px solid var(--goa-green)",
                borderRadius: "10px",
                padding: "10px 14px",
                color: "#FFFFFF",
                fontSize: "1rem",
                outline: "none",
              }}
            />
            <button
              type="button"
              className="btn-primary"
              onClick={() => {
                setIsEditingTranscript(false);
                if (onEditQuery && editedTranscript.trim()) {
                  onEditQuery(editedTranscript.trim());
                }
              }}
            >
              Ask
            </button>
            <button
              type="button"
              className="btn-secondary"
              onClick={() => setIsEditingTranscript(false)}
            >
              Cancel
            </button>
          </div>
        ) : (
          <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", gap: "14px" }}>
            <p style={{ fontSize: "1.18rem", fontWeight: 600, color: "var(--text-primary)", lineHeight: 1.45, letterSpacing: "-0.01em" }}>
              &ldquo;{data.transcriptText || data.query}&rdquo;
            </p>
            {onEditQuery && (
              <button
                type="button"
                onClick={() => {
                  setEditedTranscript(data.transcriptText || data.query);
                  setIsEditingTranscript(true);
                }}
                style={{
                  background: "rgba(255, 255, 255, 0.04)",
                  border: "1px solid var(--border-subtle)",
                  borderRadius: "8px",
                  color: "var(--tropical-green)",
                  fontSize: "0.8rem",
                  fontWeight: 600,
                  cursor: "pointer",
                  padding: "5px 10px",
                  whiteSpace: "nowrap",
                  transition: "all 0.2s ease",
                }}
                onMouseEnter={(e) => (e.currentTarget.style.borderColor = "var(--goa-green)")}
                onMouseLeave={(e) => (e.currentTarget.style.borderColor = "var(--border-subtle)")}
              >
                ✏️ Edit
              </button>
            )}
          </div>
        )}
      </div>

      {/* 2. Primary Grounded Answer Panel */}
      {data.answer && (
        <div
          className="glass-panel-elevated"
          style={{
            padding: "28px",
            border: `1px solid ${cardStyle.borderColor}`,
            boxShadow: cardStyle.glow,
            background: "rgba(21, 28, 40, 0.9)",
            borderRadius: "22px",
          }}
        >
          {/* Header */}
          <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: "18px", borderBottom: "1px solid var(--border-subtle)", paddingBottom: "14px" }}>
            <div style={{ display: "flex", alignItems: "center", gap: "10px" }}>
              <span className="font-display" style={{ fontSize: "0.85rem", fontWeight: 800, letterSpacing: "0.08em", color: "var(--text-primary)", textTransform: "uppercase" }}>
                ANSWER
              </span>
              <span
                className="badge"
                style={{
                  background: cardStyle.badgeBg,
                  color: cardStyle.badgeColor,
                  border: `1px solid ${cardStyle.borderColor}`,
                  fontSize: "0.72rem",
                  fontWeight: 700,
                }}
              >
                CONFIDENCE: {data.grounded ? confidence : "REFUSED"}
              </span>
            </div>

            <button
              type="button"
              onClick={handleCopy}
              className="btn-secondary"
              style={{
                color: copied ? "var(--tropical-green)" : "var(--text-secondary)",
                borderColor: copied ? "var(--goa-green)" : "var(--border-subtle)",
                fontSize: "0.78rem",
                padding: "5px 12px",
              }}
            >
              {copied ? "✓ Copied" : "📋 Copy"}
            </button>
          </div>

          {/* Generated Answer Content */}
          <div style={{ fontSize: "1.08rem", color: "var(--text-primary)", lineHeight: 1.7, whiteSpace: "pre-wrap", marginBottom: "22px", fontWeight: 450 }}>
            {data.answer}
          </div>

          {/* Grounding & Verification Trust Indicator */}
          <div
            style={{
              display: "flex",
              alignItems: "center",
              justifyContent: "space-between",
              flexWrap: "wrap",
              gap: "12px",
              padding: "14px 18px",
              borderRadius: "12px",
              background: data.grounded ? "rgba(0, 168, 120, 0.08)" : "rgba(255, 79, 129, 0.08)",
              border: data.grounded ? "1px solid rgba(0, 168, 120, 0.25)" : "1px solid rgba(255, 79, 129, 0.25)",
            }}
          >
            <div style={{ display: "flex", alignItems: "center", gap: "9px" }}>
              <span style={{ fontSize: "1.1rem", color: data.grounded ? "var(--tropical-green)" : "var(--pink-accent)" }}>
                {data.grounded ? "✓" : "✕"}
              </span>
              <span style={{ fontSize: "0.88rem", fontWeight: 600, color: data.grounded ? "var(--tropical-green)" : "var(--pink-accent)" }}>
                {getGroundingText()}
              </span>
            </div>

            {/* Context Explorer Toggle Button */}
            {data.sources && data.sources.length > 0 && (
              <button
                type="button"
                onClick={() => setShowContext(!showContext)}
                style={{
                  background: "transparent",
                  border: "none",
                  color: "var(--tropical-green)",
                  fontSize: "0.82rem",
                  fontWeight: 700,
                  cursor: "pointer",
                  display: "flex",
                  alignItems: "center",
                  gap: "5px",
                }}
              >
                {showContext ? "Hide Context ▲" : `Retrieved Passages (${data.sources.length}) ▼`}
              </button>
            )}
          </div>

          {/* 3. Collapsible Retrieved Passages Explorer */}
          {showContext && data.sources && data.sources.length > 0 && (
            <div style={{ marginTop: "22px", display: "flex", flexDirection: "column", gap: "12px" }}>
              <div style={{ fontSize: "0.75rem", fontWeight: 800, letterSpacing: "0.08em", color: "var(--text-secondary)", textTransform: "uppercase" }}>
                RETRIEVED EVIDENCE PASSAGES ({data.sources.length})
              </div>

              {data.sources.map((src, i) => {
                const isExpanded = expandedSources[i];
                const text = src.text || "";
                const isLong = text.length > 200;
                const displayText = isLong && !isExpanded ? text.slice(0, 200) + "..." : text;

                return (
                  <div
                    key={i}
                    style={{
                      background: "rgba(13, 17, 24, 0.75)",
                      borderRadius: "12px",
                      border: "1px solid var(--border-subtle)",
                      padding: "14px 16px",
                      fontSize: "0.9rem",
                    }}
                  >
                    <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: "8px", flexWrap: "wrap", gap: "6px" }}>
                      <div style={{ display: "flex", alignItems: "center", gap: "8px" }}>
                        <span style={{ fontSize: "0.75rem", fontWeight: 800, color: "var(--goa-green)" }}>
                          CHUNK #{i + 1}
                        </span>
                        {src.strategy && (
                          <span className="badge badge-lang" style={{ fontSize: "0.68rem" }}>
                            {src.strategy.toUpperCase()}
                          </span>
                        )}
                        {src.language && (
                          <span className="badge badge-lang" style={{ fontSize: "0.68rem" }}>
                            {src.language.toUpperCase()}
                          </span>
                        )}
                      </div>

                      {src.relevance !== undefined && (
                        <span style={{ fontFamily: "var(--font-mono)", fontSize: "0.75rem", color: "var(--ocean-blue)", fontWeight: 600 }}>
                          RETRIEVAL SCORE: {src.relevance}
                        </span>
                      )}
                    </div>

                    <p style={{ color: "#CBD5E1", lineHeight: 1.55, fontSize: "0.88rem" }}>
                      {displayText}
                    </p>

                    {isLong && (
                      <button
                        type="button"
                        onClick={() => toggleSourceExpand(i)}
                        style={{
                          background: "transparent",
                          border: "none",
                          color: "var(--tropical-green)",
                          fontSize: "0.78rem",
                          fontWeight: 700,
                          cursor: "pointer",
                          marginTop: "8px",
                          padding: "0",
                        }}
                      >
                        {isExpanded ? "Show Less ▲" : "Read Full Snippet ▼"}
                      </button>
                    )}
                  </div>
                );
              })}
            </div>
          )}
        </div>
      )}
    </div>
  );
}
