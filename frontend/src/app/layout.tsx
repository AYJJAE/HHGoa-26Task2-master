import type { Metadata } from "next";
import React from "react";
import "./globals.css";

export const metadata: Metadata = {
  title: "HH Goa 2026 — Multilingual Voice-Enabled RAG",
  description: "Voice-enabled multilingual Retrieval-Augmented Generation system supporting English, Hindi, and Marathi. Powered by ElevenLabs primary STT, Sarvam fallback, BGE-M3 hybrid search, and Gemini LLM.",
  keywords: ["RAG", "multilingual", "voice", "Hindi", "Marathi", "ElevenLabs", "Sarvam", "BGE-M3", "HH Goa"],
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" className="h-full antialiased">
      <body className="min-h-full flex flex-col">{children}</body>
    </html>
  );
}
