import type { Metadata } from "next";
import localFont from "next/font/local";
import "./globals.css";

const londrina = localFont({
  src: "../../public/fonts/LondrinaSolid.otf",
  variable: "--font-londrina",
});

export const metadata: Metadata = {
  title: "HH Goa 2026 — Multilingual Voice RAG",
  description: "Voice-enabled multilingual Retrieval-Augmented Generation system supporting Hindi, Marathi, and English. Powered by BGE-M3 hybrid search, Gemini LLM, and Sarvam STT.",
  keywords: ["RAG", "multilingual", "voice", "Hindi", "Marathi", "search", "AI", "HH Goa"],
};

export default function RootLayout({ children }: LayoutProps<"/">) {
  return (
    <html
      lang="en"
      className={`${londrina.variable} h-full antialiased`}
    >
      <body className="min-h-full flex flex-col">{children}</body>
    </html>
  );
}
