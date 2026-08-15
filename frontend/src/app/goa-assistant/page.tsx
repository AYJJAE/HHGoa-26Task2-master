"use client";

import Link from "next/link";
import Image from "next/image";
import "../globals.css";

export default function GoaAssistantPage() {
  return (
    <div className="home-viewport">
      {/* Fullscreen Video Background */}
      <video
        autoPlay
        muted
        loop
        playsInline
        className="home-bg-video"
        poster="/HHGoa-1.png"
      >
        <source src="/homepage-video.mp4" type="video/mp4" />
        <source src="/homepage%20video.mp4" type="video/mp4" />
      </video>

      {/* Tropical Green / Dark Translucent Overlay */}
      <div className="home-overlay" />

      {/* Content Container */}
      <div className="home-content" style={{ maxWidth: "880px" }}>
        {/* Top Header */}
        <header className="home-header" style={{ marginBottom: "2rem" }}>
          <Link href="/" className="logo-link">
            <Image
              src="/HHGoa-1.png"
              alt="Hacker House Goa Logo"
              width={160}
              height={70}
              priority
              className="logo-img"
              style={{ objectFit: "contain" }}
            />
          </Link>
          <div className="horizon-badge-container">
            <Image
              src="/horizonlogo.png"
              alt="Horizon Logo"
              width={190}
              height={85}
              priority
              className="horizon-logo-img"
              style={{ objectFit: "contain" }}
            />
          </div>
        </header>

        {/* Main Card */}
        <div 
          className="glass-container" 
          style={{ 
            background: "rgba(6, 26, 18, 0.88)", 
            borderColor: "rgba(255, 230, 0, 0.35)", 
            padding: "40px",
            boxShadow: "0 20px 60px rgba(0,0,0,0.6), 0 0 40px rgba(255, 230, 0, 0.15)",
            borderRadius: "28px"
          }}
        >
          {/* Badge */}
          <div style={{ display: "flex", gap: "10px", alignItems: "center", marginBottom: "20px" }}>
            <span style={{
              background: "rgba(255, 19, 125, 0.2)",
              color: "#FF137D",
              border: "1px solid rgba(255, 19, 125, 0.5)",
              padding: "6px 14px",
              borderRadius: "9999px",
              fontSize: "0.8rem",
              fontWeight: 800,
              letterSpacing: "0.08em",
              textTransform: "uppercase"
            }}>
              ✨ Feature Coming Next
            </span>
            <span style={{ color: "#FFE600", fontSize: "0.85rem", fontWeight: 600 }}>
              🌴 Vagator • North Goa
            </span>
          </div>

          <h1 style={{ 
            fontFamily: "var(--font-display)", 
            fontSize: "clamp(2rem, 5vw, 3.2rem)", 
            fontWeight: 900, 
            color: "#FFE600",
            lineHeight: 1.1,
            marginBottom: "16px",
            textTransform: "uppercase",
            letterSpacing: "-0.02em"
          }}>
            Goa Assistant
          </h1>

          <p style={{ 
            fontSize: "1.2rem", 
            color: "#E2E8F0", 
            lineHeight: 1.6,
            marginBottom: "28px",
            maxWidth: "680px"
          }}>
            Your dedicated AI companion for exploring the island — discovering sunset hackathons, beach co-working spots, Konkani dialect insights, and local creative-tech gatherings across Goa.
          </p>

          {/* Feature Teasers Grid */}
          <div style={{ 
            display: "grid", 
            gridTemplateColumns: "repeat(auto-fit, minmax(200px, 1fr))", 
            gap: "16px",
            marginBottom: "36px"
          }}>
            <div style={{
              background: "rgba(255, 255, 255, 0.04)",
              border: "1px solid rgba(255, 255, 255, 0.08)",
              padding: "16px",
              borderRadius: "16px"
            }}>
              <span style={{ fontSize: "1.5rem" }}>🥥</span>
              <h4 style={{ color: "#FFE600", fontSize: "0.95rem", marginTop: "8px", marginBottom: "4px" }}>Local Culture & Places</h4>
              <p style={{ color: "#94A3B8", fontSize: "0.8rem" }}>Hidden beaches, local shacks, and historic Portuguese architecture.</p>
            </div>

            <div style={{
              background: "rgba(255, 255, 255, 0.04)",
              border: "1px solid rgba(255, 255, 255, 0.08)",
              padding: "16px",
              borderRadius: "16px"
            }}>
              <span style={{ fontSize: "1.5rem" }}>💻</span>
              <h4 style={{ color: "#FF137D", fontSize: "0.95rem", marginTop: "8px", marginBottom: "4px" }}>Hacker House Hubs</h4>
              <p style={{ color: "#94A3B8", fontSize: "0.8rem" }}>Real-time event schedules, side-events, and co-working spaces.</p>
            </div>

            <div style={{
              background: "rgba(255, 255, 255, 0.04)",
              border: "1px solid rgba(255, 255, 255, 0.08)",
              padding: "16px",
              borderRadius: "16px"
            }}>
              <span style={{ fontSize: "1.5rem" }}>🗣️</span>
              <h4 style={{ color: "#10B981", fontSize: "0.95rem", marginTop: "8px", marginBottom: "4px" }}>Konkani & Multilingual</h4>
              <p style={{ color: "#94A3B8", fontSize: "0.8rem" }}>Native Konkani voice synthesis and translation assistance.</p>
            </div>
          </div>

          {/* Action Buttons */}
          <div style={{ display: "flex", gap: "16px", flexWrap: "wrap", alignItems: "center" }}>
            <Link
              href="/"
              className="destination-card-btn"
              style={{
                background: "rgba(255, 255, 255, 0.1)",
                color: "#FFFFFF",
                border: "1px solid rgba(255, 255, 255, 0.2)",
                padding: "12px 24px",
                borderRadius: "14px",
                textDecoration: "none",
                fontWeight: 700,
                fontSize: "0.95rem",
                display: "inline-flex",
                alignItems: "center",
                gap: "8px",
                transition: "all 0.2s ease"
              }}
            >
              ← Return to Home
            </Link>

            <Link
              href="/rag"
              className="destination-card-btn"
              style={{
                background: "linear-gradient(135deg, #FF137D, #FF0055)",
                color: "#FFFFFF",
                padding: "12px 28px",
                borderRadius: "14px",
                textDecoration: "none",
                fontWeight: 800,
                fontSize: "0.95rem",
                display: "inline-flex",
                alignItems: "center",
                gap: "8px",
                boxShadow: "0 8px 24px rgba(255, 19, 125, 0.4)",
                transition: "all 0.2s ease"
              }}
            >
              Try Live Multilingual RAG →
            </Link>
          </div>
        </div>

        {/* Footer */}
        <footer className="home-footer">
          <p>© 2026 Hacker House Goa. Built with ⚡ in Vagator, Goa.</p>
        </footer>
      </div>
    </div>
  );
}
