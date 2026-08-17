"use client";

import Link from "next/link";
import GoaMapBackground from "../components/GoaMapBackground";
import "../globals.css";

export default function GoaAssistantPage() {
  return (
    <div style={{ position: "relative", minHeight: "100vh", display: "flex", flexDirection: "column", backgroundColor: "var(--bg-primary)" }}>
      {/* Goa Digital Map Background */}
      <GoaMapBackground />

      {/* Floating Glass Navigation */}
      <header
        style={{
          position: "sticky",
          top: "16px",
          zIndex: 30,
          maxWidth: "980px",
          width: "calc(100% - 32px)",
          margin: "0 auto",
          backdropFilter: "blur(20px)",
          WebkitBackdropFilter: "blur(20px)",
          border: "1px solid rgba(255, 255, 255, 0.08)",
          background: "rgba(13, 17, 24, 0.85)",
          borderRadius: "9999px",
          padding: "10px 22px",
          boxShadow: "0 10px 30px rgba(0, 0, 0, 0.5)",
        }}
      >
        <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", flexWrap: "wrap", gap: "12px" }}>
          <Link href="/" style={{ textDecoration: "none", display: "flex", alignItems: "center", gap: "10px" }}>
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
            <span className="font-display" style={{ fontSize: "0.95rem", fontWeight: 800, color: "#F5F7FA" }}>
              Voice RAG Goa
            </span>
          </Link>

          <Link href="/rag" className="btn-primary" style={{ padding: "6px 16px", fontSize: "0.82rem", textDecoration: "none" }}>
            Launch Voice RAG →
          </Link>
        </div>
      </header>

      {/* Main Content Container */}
      <main
        style={{
          position: "relative",
          zIndex: 10,
          flex: 1,
          maxWidth: "880px",
          width: "100%",
          margin: "0 auto",
          padding: "40px 20px 60px 20px",
          display: "flex",
          flexDirection: "column",
          gap: "28px",
        }}
      >
        {/* Feature Hero Card */}
        <div 
          className="glass-panel-elevated" 
          style={{ 
            padding: "40px 32px",
            background: "rgba(21, 28, 40, 0.9)",
            border: "1px solid rgba(0, 168, 120, 0.35)",
            boxShadow: "0 20px 50px rgba(0,0,0,0.6), 0 0 40px rgba(0, 168, 120, 0.12)",
          }}
        >
          {/* Badge */}
          <div style={{ display: "flex", gap: "10px", alignItems: "center", marginBottom: "20px", flexWrap: "wrap" }}>
            <span style={{
              background: "rgba(255, 79, 129, 0.15)",
              color: "var(--pink-accent)",
              border: "1px solid rgba(255, 79, 129, 0.4)",
              padding: "4px 12px",
              borderRadius: "9999px",
              fontSize: "0.75rem",
              fontWeight: 800,
              letterSpacing: "0.08em",
              textTransform: "uppercase"
            }}>
              ✨ Goa Coastal Companion
            </span>
            <span className="badge badge-node" style={{ fontSize: "0.72rem" }}>
              🌴 Vagator Node • 15.60° N, 73.74° E
            </span>
          </div>

          <h1 className="font-display" style={{ 
            fontSize: "clamp(2.2rem, 5vw, 3.4rem)", 
            fontWeight: 800, 
            color: "var(--text-primary)",
            lineHeight: 1.15,
            marginBottom: "16px",
            letterSpacing: "-0.02em"
          }}>
            Goa <span className="gradient-goa-accent">Assistant</span>
          </h1>

          <p style={{ 
            fontSize: "1.1rem", 
            color: "var(--text-secondary)", 
            lineHeight: 1.65,
            marginBottom: "32px",
            maxWidth: "680px"
          }}>
            Your dedicated AI companion for exploring the state — discovering sunset hackathons, beach co-working spots, Konkani dialect insights, and local creative-tech gatherings across Goa.
          </p>

          {/* Feature Grid */}
          <div style={{ 
            display: "grid", 
            gridTemplateColumns: "repeat(auto-fit, minmax(220px, 1fr))", 
            gap: "16px",
            marginBottom: "36px"
          }}>
            <div style={{
              background: "rgba(13, 17, 24, 0.8)",
              border: "1px solid var(--border-subtle)",
              padding: "20px",
              borderRadius: "16px"
            }}>
              <span style={{ fontSize: "1.6rem" }}>🥥</span>
              <h4 style={{ color: "var(--tropical-green)", fontSize: "1rem", marginTop: "10px", marginBottom: "6px", fontWeight: 700 }}>Local Culture & Places</h4>
              <p style={{ color: "var(--text-muted)", fontSize: "0.85rem", lineHeight: 1.5 }}>Historic Portuguese architecture, local shacks, and coastal pathways.</p>
            </div>

            <div style={{
              background: "rgba(13, 17, 24, 0.8)",
              border: "1px solid var(--border-subtle)",
              padding: "20px",
              borderRadius: "16px"
            }}>
              <span style={{ fontSize: "1.6rem" }}>💻</span>
              <h4 style={{ color: "var(--pink-accent)", fontSize: "1rem", marginTop: "10px", marginBottom: "6px", fontWeight: 700 }}>Hacker House Hubs</h4>
              <p style={{ color: "var(--text-muted)", fontSize: "0.85rem", lineHeight: 1.5 }}>Real-time hackathon side-events, developer schedules, and co-working hubs.</p>
            </div>

            <div style={{
              background: "rgba(13, 17, 24, 0.8)",
              border: "1px solid var(--border-subtle)",
              padding: "20px",
              borderRadius: "16px"
            }}>
              <span style={{ fontSize: "1.6rem" }}>🗣️</span>
              <h4 style={{ color: "var(--ocean-blue)", fontSize: "1rem", marginTop: "10px", marginBottom: "6px", fontWeight: 700 }}>Native Konkani & Multilingual</h4>
              <p style={{ color: "var(--text-muted)", fontSize: "0.85rem", lineHeight: 1.5 }}>Native Konkani, Marathi, Hindi, and English voice synthesis and translation.</p>
            </div>
          </div>

          {/* Action Buttons */}
          <div style={{ display: "flex", gap: "14px", flexWrap: "wrap", alignItems: "center" }}>
            <Link
              href="/"
              className="btn-primary"
              style={{ textDecoration: "none", padding: "12px 24px" }}
            >
              ← Back to Main Voice RAG
            </Link>

            <Link
              href="/rag"
              className="btn-secondary"
              style={{ textDecoration: "none", padding: "12px 22px" }}
            >
              Try Live Voice Queries →
            </Link>
          </div>
        </div>
      </main>

      {/* Footer */}
      <footer
        style={{
          borderTop: "1px solid var(--border-subtle)",
          padding: "24px 20px",
          textAlign: "center",
          color: "var(--text-muted)",
          fontSize: "0.82rem",
          background: "rgba(8, 10, 15, 0.8)",
          position: "relative",
          zIndex: 10,
        }}
      >
        <p>© 2026 Hacker House Goa. Built with ⚡ in Vagator, Goa [15.60° N, 73.74° E].</p>
      </footer>
    </div>
  );
}
