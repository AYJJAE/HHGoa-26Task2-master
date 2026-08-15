"use client";

import Link from "next/link";
import Image from "next/image";
import AnimatedContent from "./components/AnimatedContent";
import GlitchText from "./components/GlitchText";
import PillNav from "./components/PillNav";
import "./globals.css";

export default function HomePage() {
  return (
    <div className="home-viewport">
      {/* Full-viewport Background Video */}
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

      {/* Tropical Deep Green Translucent Overlay */}
      <div className="home-overlay" />

      {/* Foreground Content Container */}
      <div className="home-content">
        {/* Top Branding Navigation */}
        <header className="home-header">
          <div className="logo-badge-container">
            <Image
              src="/HHGoa-1.png"
              alt="Hacker House Goa Logo"
              width={190}
              height={85}
              priority
              className="logo-img"
              style={{ objectFit: "contain" }}
            />
          </div>

          <PillNav
            items={[
              { label: 'About', href: '#about' },
              { label: 'Demo Video', href: '#demo' },
              { label: 'Team', href: '#team' },
              { label: 'Task 1', href: 'https://hh-goa-26.vercel.app/' }
            ]}
            activeHref=""
            className="home-pill-nav"
            ease="power2.easeOut"
            baseColor="rgba(6, 26, 18, 0.65)"
            pillColor="#FF137D"
            hoveredPillTextColor="#FFFFFF"
            pillTextColor="#FFE600"
          />

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

        {/* Hero Section */}
        <section className="home-hero">
          <div className="eyebrow-tag">BUILT FOR HACKER HOUSE GOA - BY TEAM HORIZON LABS</div>
          <h1 className="home-title">
            <GlitchText
              speed={1}
              enableShadows={true}
              enableOnHover={true}
              className="home-glitch-text"
            >
              INTELLIGENCE IN PARADISE
            </GlitchText>
          </h1>
          <p className="home-subtitle">
            A unique take on Task two for HH GOA where we do not just build a RAG pipeline we build something meaningful with it
          </p>
        </section>

        {/* Two Staggered Destination Cards / Signage */}
        <div className="destination-grid">
          {/* Destination Card 1: MULTILINGUAL RAG */}
          <AnimatedContent
            distance={150}
            direction="horizontal"
            reverse={false}
            duration={1.2}
            ease="bounce.out"
            initialOpacity={0.2}
            animateOpacity
            scale={1.1}
            threshold={0.2}
            delay={0.25}
            className="destination-anim-wrapper"
          >
            <Link href="/rag" className="destination-card card-rag">
              <div className="card-top-tag">
                <span className="card-badge rag-badge">LIVE PRODUCTION</span>
                <span className="card-lang-tag">HI • MR • EN</span>
              </div>

              <div className="card-body">
                <div className="card-icon-circle rag-icon-circle">
                  <span>🎙️</span>
                </div>
                <h2 className="card-title">MULTILINGUAL RAG</h2>
                <p className="card-description">
                  Voice-first multilingual knowledge retrieval
                </p>
              </div>

              <div className="card-footer">
                <span className="card-cta rag-cta">
                  Enter the live RAG experience
                  <span className="arrow-icon">→</span>
                </span>
              </div>
            </Link>
          </AnimatedContent>

          {/* Destination Card 2: GOA ASSISTANT */}
          <AnimatedContent
            distance={150}
            direction="horizontal"
            reverse={true}
            duration={1.2}
            ease="bounce.out"
            initialOpacity={0.2}
            animateOpacity
            scale={1.1}
            threshold={0.2}
            delay={0.45}
            className="destination-anim-wrapper"
          >
            <Link href="/goa-assistant" className="destination-card card-assistant">
              <div className="card-top-tag">
                <span className="card-badge assistant-badge">COMING NEXT</span>
                <span className="card-lang-tag">ISLAND COMPANION</span>
              </div>

              <div className="card-body">
                <div className="card-icon-circle assistant-icon-circle">
                  <span>🌴</span>
                </div>
                <h2 className="card-title">GOA ASSISTANT</h2>
                <p className="card-description">
                  Your AI companion for Goa
                </p>
              </div>

              <div className="card-footer">
                <span className="card-cta assistant-cta">
                  Explore the island with AI
                  <span className="arrow-icon">→</span>
                </span>
              </div>
            </Link>
          </AnimatedContent>
        </div>

        {/* Footer info */}
        <footer className="home-footer">
          <p>
            Hacker House Goa 2026 • Powered by Hybrid BGE-M3 Retrieval, Gemini LLM & Sarvam STT
          </p>
        </footer>
      </div>
    </div>
  );
}
