"use client";

import React from "react";

/**
 * GoaMapBackground
 * Extremely subtle, lightweight SVG/CSS digital navigation map of Goa at night.
 * Features abstract coastal curves, road route lines, coordinates, and glowing nodes.
 * Rendered at very low opacity (0.03 - 0.08) to remain elegant and non-distracting.
 */
export default function GoaMapBackground() {
  return (
    <div
      aria-hidden="true"
      style={{
        position: "fixed",
        inset: 0,
        pointerEvents: "none",
        zIndex: 0,
        overflow: "hidden",
        backgroundColor: "#080A0F",
      }}
    >
      {/* 1. Deep Ambient Radial Gradients (Goa Nightglow) */}
      <div
        style={{
          position: "absolute",
          top: "-10%",
          left: "20%",
          width: "700px",
          height: "600px",
          background: "radial-gradient(ellipse, rgba(0, 168, 120, 0.08) 0%, rgba(22, 199, 132, 0.03) 40%, transparent 70%)",
          filter: "blur(90px)",
        }}
      />
      <div
        style={{
          position: "absolute",
          top: "30%",
          right: "-5%",
          width: "650px",
          height: "550px",
          background: "radial-gradient(ellipse, rgba(255, 122, 61, 0.06) 0%, rgba(255, 79, 129, 0.03) 45%, transparent 70%)",
          filter: "blur(100px)",
        }}
      />
      <div
        style={{
          position: "absolute",
          bottom: "-10%",
          left: "35%",
          width: "800px",
          height: "500px",
          background: "radial-gradient(ellipse, rgba(22, 138, 173, 0.07) 0%, transparent 65%)",
          filter: "blur(110px)",
        }}
      />

      {/* 2. Abstract Goa Digital Vector Map (Coastline, Routes, Radar Rings) */}
      <svg
        width="100%"
        height="100%"
        viewBox="0 0 1440 900"
        preserveAspectRatio="xMidYMid slice"
        style={{ position: "absolute", inset: 0, width: "100%", height: "100%" }}
        fill="none"
        xmlns="http://www.w3.org/2000/svg"
      >
        <defs>
          {/* Subtle Grid Pattern */}
          <pattern id="goa-grid" width="80" height="80" patternUnits="userSpaceOnUse">
            <path
              d="M 80 0 L 0 0 0 80"
              fill="none"
              stroke="rgba(255, 255, 255, 0.02)"
              strokeWidth="1"
            />
            <circle cx="80" cy="80" r="1" fill="rgba(0, 168, 120, 0.15)" />
          </pattern>

          {/* Glowing Gradients */}
          <linearGradient id="route-green" x1="0%" y1="0%" x2="100%" y2="100%">
            <stop offset="0%" stopColor="#00A878" stopOpacity="0.18" />
            <stop offset="50%" stopColor="#16C784" stopOpacity="0.12" />
            <stop offset="100%" stopColor="#00A878" stopOpacity="0.04" />
          </linearGradient>

          <linearGradient id="route-orange" x1="0%" y1="100%" x2="100%" y2="0%">
            <stop offset="0%" stopColor="#FF7A3D" stopOpacity="0.14" />
            <stop offset="100%" stopColor="#FFC857" stopOpacity="0.03" />
          </linearGradient>

          <linearGradient id="coast-line" x1="0%" y1="0%" x2="0%" y2="100%">
            <stop offset="0%" stopColor="#168AAD" stopOpacity="0.15" />
            <stop offset="40%" stopColor="#00A878" stopOpacity="0.2" />
            <stop offset="70%" stopColor="#FF7A3D" stopOpacity="0.15" />
            <stop offset="100%" stopColor="#168AAD" stopOpacity="0.08" />
          </linearGradient>
        </defs>

        {/* Global Grid Overlay */}
        <rect width="100%" height="100%" fill="url(#goa-grid)" />

        {/* --- Abstract Arabian Sea & Goa Coastline Curves (West Edge) --- */}
        <path
          d="M 180,-20 Q 240,160 210,320 T 270,520 T 230,750 T 290,950"
          stroke="url(#coast-line)"
          strokeWidth="2.5"
          fill="none"
          strokeDasharray="8 6"
        />
        <path
          d="M 140,-20 Q 200,180 170,340 T 230,540 T 190,770 T 250,950"
          stroke="rgba(22, 138, 173, 0.08)"
          strokeWidth="1.5"
          fill="none"
        />

        {/* --- Mandovi & Zuari River Estuary Curves --- */}
        {/* Mandovi River Curve (Connecting to Panaji) */}
        <path
          d="M 230,420 C 380,410 520,440 680,410 T 940,430 T 1200,390"
          stroke="rgba(0, 168, 120, 0.1)"
          strokeWidth="1.8"
          fill="none"
        />
        {/* Zuari River Curve (Connecting to Vasco / South) */}
        <path
          d="M 250,560 C 420,580 580,550 760,570 T 1080,540"
          stroke="rgba(22, 138, 173, 0.09)"
          strokeWidth="1.5"
          fill="none"
        />

        {/* --- Highway Route Lines (NH66 & Coast Road) --- */}
        <path
          d="M 360,-40 Q 340,220 380,380 T 350,600 T 400,820 T 370,950"
          stroke="url(#route-green)"
          strokeWidth="2"
          fill="none"
        />
        <path
          d="M 480,-40 L 490,260 L 640,420 L 620,700 L 780,950"
          stroke="url(#route-orange)"
          strokeWidth="1.5"
          fill="none"
          strokeDasharray="4 8"
        />

        {/* --- Digital Radar Location Nodes (Key Goa Hubs) --- */}

        {/* Node 1: Vagator / Hacker House Node (North Coast • 15.60° N, 73.74° E) */}
        <g transform="translate(240, 240)">
          <circle cx="0" cy="0" r="28" fill="none" stroke="rgba(0, 168, 120, 0.12)" strokeWidth="1" strokeDasharray="3 3" />
          <circle cx="0" cy="0" r="14" fill="none" stroke="rgba(0, 168, 120, 0.25)" strokeWidth="1.2" />
          <circle cx="0" cy="0" r="4" fill="#00A878" opacity="0.85" />
          <circle cx="0" cy="0" r="2" fill="#FFFFFF" />
          <text x="18" y="4" fill="rgba(0, 168, 120, 0.65)" fontSize="10" fontFamily="var(--font-mono)" letterSpacing="0.08em">
            VAGATOR [15.60°N, 73.74°E]
          </text>
        </g>

        {/* Node 2: Panaji Capital Node (15.49° N, 73.82° E) */}
        <g transform="translate(360, 420)">
          <circle cx="0" cy="0" r="22" fill="none" stroke="rgba(255, 200, 87, 0.12)" strokeWidth="1" />
          <circle cx="0" cy="0" r="10" fill="none" stroke="rgba(255, 200, 87, 0.25)" strokeWidth="1" />
          <circle cx="0" cy="0" r="3.5" fill="#FFC857" opacity="0.8" />
          <text x="16" y="4" fill="rgba(255, 200, 87, 0.55)" fontSize="9" fontFamily="var(--font-mono)" letterSpacing="0.06em">
            PANAJI CORE [15.49°N, 73.82°E]
          </text>
        </g>

        {/* Node 3: Calangute / Coastal Edge (15.54° N, 73.75° E) */}
        <g transform="translate(230, 340)">
          <circle cx="0" cy="0" r="6" fill="none" stroke="rgba(255, 79, 129, 0.3)" strokeWidth="1" />
          <circle cx="0" cy="0" r="2.5" fill="#FF4F81" opacity="0.75" />
        </g>

        {/* Node 4: Margao South Hub (15.27° N, 73.96° E) */}
        <g transform="translate(390, 720)">
          <circle cx="0" cy="0" r="20" fill="none" stroke="rgba(255, 122, 61, 0.15)" strokeWidth="1" strokeDasharray="2 4" />
          <circle cx="0" cy="0" r="3.5" fill="#FF7A3D" opacity="0.8" />
          <text x="16" y="4" fill="rgba(255, 122, 61, 0.5)" fontSize="9" fontFamily="var(--font-mono)" letterSpacing="0.06em">
            MARGAO [15.27°N, 73.96°E]
          </text>
        </g>

        {/* Subtle Map Coordinates / Compass Crosshairs */}
        <g transform="translate(1320, 80)" opacity="0.3">
          <circle cx="0" cy="0" r="24" stroke="rgba(255,255,255,0.15)" strokeWidth="1" />
          <line x1="-30" y1="0" x2="30" y2="0" stroke="rgba(255,255,255,0.15)" strokeWidth="1" />
          <line x1="0" y1="-30" x2="0" y2="30" stroke="rgba(255,255,255,0.15)" strokeWidth="1" />
          <text x="-4" y="-12" fill="rgba(255,255,255,0.4)" fontSize="8" fontFamily="var(--font-mono)">N</text>
          <text x="6" y="16" fill="rgba(0,168,120,0.6)" fontSize="7" fontFamily="var(--font-mono)">GOA-GRID</text>
        </g>
      </svg>
    </div>
  );
}
