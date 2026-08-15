"use client";

import { useState, useRef, useCallback, useEffect } from "react";
import dynamic from "next/dynamic";
import Link from "next/link";
import "../globals.css";

const Strands = dynamic(() => import("../components/Strands"), { ssr: false });

// Interface for API Response matching FastAPI
interface RAGResponse {
  status: string;
  query: string;
  answer?: string;
  message?: string;
  sources?: any[];
  routing?: {
    language: string;
    intent: string;
    complexity: string;
    strategy?: any;
  };
  retrieval_confidence?: string;
  grounded?: boolean;
  grounding?: {
    status: string;
    reason?: string;
  };
  context_sufficient?: boolean;
  refusal_reason?: string;
  latency_metrics?: Record<string, number>;
}

// Toast notification types
interface Toast {
  id: string;
  type: "error" | "warning" | "success" | "info";
  message: string;
  dismissing?: boolean;
}

// Intent icon/color map
const INTENT_ICONS: Record<string, string> = {
  "Chitchat": "💬",
  "Factual": "🔍",
  "Entity-heavy": "🏷️",
  "Comparative": "⚖️",
  "Multi-hop": "🔗",
};

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

export default function RAGPage() {
  const [isRecording, setIsRecording] = useState(false);
  const [messages, setMessages] = useState<{role: "user" | "agent", content: string | RAGResponse}[]>([]);
  const [isProcessing, setIsProcessing] = useState(false);
  const [inputMode, setInputMode] = useState<"voice" | "text">("voice");
  const [textQuery, setTextQuery] = useState("");
  const [toasts, setToasts] = useState<Toast[]>([]);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const chunksRef = useRef<BlobPart[]>([]);
  const chatEndRef = useRef<HTMLDivElement>(null);
  const currentRequestIdRef = useRef<number>(0);

  // Auto-scroll to bottom when messages change
  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, isProcessing]);

  // --- Toast System ---
  const addToast = useCallback((type: Toast["type"], message: string) => {
    const id = Date.now().toString() + Math.random().toString(36).slice(2);
    setToasts(prev => [...prev, { id, type, message }]);
    // Auto-dismiss after 5 seconds
    setTimeout(() => {
      setToasts(prev => prev.map(t => t.id === id ? { ...t, dismissing: true } : t));
      setTimeout(() => {
        setToasts(prev => prev.filter(t => t.id !== id));
      }, 300);
    }, 5000);
  }, []);

  const toastIcons: Record<string, string> = {
    error: "✕",
    warning: "⚠",
    success: "✓",
    info: "ℹ",
  };

  // --- Voice Recording ---
  const startRecording = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const mediaRecorder = new MediaRecorder(stream);
      mediaRecorderRef.current = mediaRecorder;
      chunksRef.current = [];

      mediaRecorder.ondataavailable = (e) => {
        if (e.data.size > 0) {
          chunksRef.current.push(e.data);
        }
      };

      mediaRecorder.onstop = async () => {
        const audioBlob = new Blob(chunksRef.current, { type: "audio/wav" });
        await sendAudio(audioBlob);
      };

      mediaRecorder.start();
      setIsRecording(true);
    } catch (error) {
      console.error("Error accessing microphone:", error);
      addToast("error", "Microphone access is required for voice input. Please allow microphone permissions.");
    }
  };

  const stopRecording = () => {
    if (mediaRecorderRef.current && isRecording) {
      mediaRecorderRef.current.stop();
      mediaRecorderRef.current.stream.getTracks().forEach(track => track.stop());
      setIsRecording(false);
    }
  };

  const sendAudio = async (audioBlob: Blob) => {
    setIsProcessing(true);
    
    const requestId = Date.now();
    currentRequestIdRef.current = requestId;
    
    // Clear previous chat, set single-turn placeholder
    setMessages([{ role: "user", content: "🎤 ..." }]);
    
    const formData = new FormData();
    formData.append("audio", audioBlob, "recording.wav");

    try {
      const response = await fetch(`${API_BASE}/api/voice_ask`, {
        method: "POST",
        body: formData,
      });
      
      if (currentRequestIdRef.current !== requestId) return; // Ignore stale
      
      if (!response.ok) {
        throw new Error(`Server returned ${response.status}`);
      }
      
      const data: RAGResponse = await response.json();
      
      if (currentRequestIdRef.current !== requestId) return; // Ignore stale
      
      setMessages([
        { role: "user", content: data.query || "Could not transcribe." },
        { role: "agent", content: data }
      ]);
      
    } catch (error: any) {
      if (currentRequestIdRef.current !== requestId) return; // Ignore stale
      console.error("API Error:", error);
      const isNetworkError = error.message?.includes("fetch") || error.message?.includes("network") || error.name === "TypeError";
      
      if (isNetworkError) {
        addToast("error", "Network disconnected. Please check your connection and try again.");
      } else {
        addToast("error", `Server error: ${error.message || "Unknown error"}`);
      }
      
      setMessages([]); // Remove the placeholder on error
    } finally {
      if (currentRequestIdRef.current === requestId) {
        setIsProcessing(false);
      }
    }
  };

  const sendTextQuery = async () => {
    const query = textQuery.trim();
    if (!query || isProcessing) return;
    
    setIsProcessing(true);
    setTextQuery("");
    
    const requestId = Date.now();
    currentRequestIdRef.current = requestId;
    
    // Clear previous chat, set single-turn query
    setMessages([{ role: "user", content: query }]);

    try {
      const response = await fetch(`${API_BASE}/api/ask`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query }),
      });
      
      if (currentRequestIdRef.current !== requestId) return; // Ignore stale
      
      if (!response.ok) {
        throw new Error(`Server returned ${response.status}`);
      }
      
      const data: RAGResponse = await response.json();
      
      if (currentRequestIdRef.current !== requestId) return; // Ignore stale
      
      setMessages([
        { role: "user", content: query },
        { role: "agent", content: data }
      ]);
      
    } catch (error: any) {
      if (currentRequestIdRef.current !== requestId) return; // Ignore stale
      console.error("API Error:", error);
      const isNetworkError = error.message?.includes("fetch") || error.message?.includes("network") || error.name === "TypeError";
      
      if (isNetworkError) {
        addToast("error", "Network disconnected. Please check your connection and try again.");
      } else {
        addToast("error", `Server error: ${error.message || "Unknown error"}`);
      }
      
      setMessages([{ role: "user", content: query }, { 
        role: "agent", 
        content: { status: "error", query, message: "Failed to connect to backend API." } 
      }]);
    } finally {
      if (currentRequestIdRef.current === requestId) {
        setIsProcessing(false);
      }
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      sendTextQuery();
    }
  };

  // --- Render Agent Message ---
  const renderAgentMessage = (data: RAGResponse) => {
    if (data.status === "error") {
      return (
        <div className="message agent" style={{ borderColor: 'var(--error-color)' }}>
          <p>{data.message}</p>
        </div>
      );
    }

    return (
      <div className="message agent glass-container" style={{ padding: '24px' }}>
        {/* Intent Badge Row */}
        {data.routing && (
          <div style={{ display: 'flex', gap: '8px', alignItems: 'center', marginBottom: '16px', flexWrap: 'wrap' }}>
            <span className={`intent-badge ${data.routing.intent}`}>
              <span className="intent-icon">{INTENT_ICONS[data.routing.intent] || "📋"}</span>
              {data.routing.intent}
            </span>
            <span style={{ fontSize: '0.75rem', color: 'var(--text-secondary)' }}>
              {data.routing.language} • {data.routing.complexity}
            </span>
          </div>
        )}

        {data.status === "answered" ? (
          <div style={{ marginBottom: '20px' }}>
            <p style={{ fontSize: '1.1rem' }}>{data.answer}</p>
          </div>
        ) : (
          <div style={{ marginBottom: '20px', color: 'var(--warning-color)' }}>
            <p>{data.answer || data.message}</p>
          </div>
        )}

        {/* Refusal / Grounding Status Bar */}
        {data.status === "answered" && data.grounded ? (
           <div className="answerability-bar sufficient">
             <span className="answerability-icon">🟢</span>
             <span><strong>GROUNDED</strong> | {data.retrieval_confidence} CONFIDENCE</span>
           </div>
        ) : data.status === "refused" ? (
           <div className="answerability-bar insufficient">
             <span className="answerability-icon">🟡</span>
             <span><strong>INSUFFICIENT CONTEXT</strong> | {data.retrieval_confidence || "LOW"} CONFIDENCE</span>
           </div>
        ) : null}
        
        {/* Detailed Metrics Dashboard */}
        <div className="metrics-grid">
          {data.latency_metrics?.total_e2e_ms && (
            <div className="metric-card" style={{ gridColumn: 'span 2', background: 'rgba(59, 130, 246, 0.15)', borderColor: 'rgba(59, 130, 246, 0.3)', border: '1px solid' }}>
              <span className="metric-value">{data.latency_metrics.total_e2e_ms.toFixed(1)}ms</span>
              <span className="metric-label">VOICE → ANSWER (Total)</span>
            </div>
          )}

          {data.latency_metrics?.total_retrieval_ms !== undefined ? (
            <div className="metric-card" style={{ gridColumn: 'span 2', background: 'rgba(139, 92, 246, 0.15)', borderColor: 'rgba(139, 92, 246, 0.3)', border: '1px solid' }}>
              <span className="metric-value">{data.latency_metrics.total_retrieval_ms.toFixed(1)}ms</span>
              <span className="metric-label">RAG RETRIEVAL (Core)</span>
            </div>
          ) : (
            <div className="metric-card" style={{ gridColumn: 'span 2', background: 'rgba(139, 92, 246, 0.15)', borderColor: 'rgba(139, 92, 246, 0.3)', border: '1px solid' }}>
               <span className="metric-value">
                 {((data.latency_metrics?.query_embedding_ms || 0) + (data.latency_metrics?.dense_retrieval_ms || 0) + (data.latency_metrics?.sparse_retrieval_ms || 0) + (data.latency_metrics?.rrf_fusion_ms || 0)).toFixed(1)}ms
               </span>
               <span className="metric-label">RAG RETRIEVAL (Core)</span>
            </div>
          )}
          
          {data.latency_metrics?.stt_ms && (
             <div className="metric-card">
               <span className="metric-value">{data.latency_metrics.stt_ms.toFixed(1)}ms</span>
               <span className="metric-label">STT</span>
             </div>
          )}
          
          {data.latency_metrics?.query_routing_ms && (
             <div className="metric-card">
               <span className="metric-value">{data.latency_metrics.query_routing_ms.toFixed(1)}ms</span>
               <span className="metric-label">Router</span>
             </div>
          )}

          {data.latency_metrics?.query_embedding_ms !== undefined && (
            <div className="metric-card">
              <span className="metric-value">{data.latency_metrics.query_embedding_ms.toFixed(1)}ms</span>
              <span className="metric-label">Embed</span>
            </div>
          )}

          {data.latency_metrics?.dense_retrieval_ms !== undefined && (
            <div className="metric-card">
              <span className="metric-value">{data.latency_metrics.dense_retrieval_ms.toFixed(1)}ms</span>
              <span className="metric-label">Dense</span>
            </div>
          )}

          {data.latency_metrics?.sparse_retrieval_ms !== undefined && (
            <div className="metric-card">
              <span className="metric-value">{data.latency_metrics.sparse_retrieval_ms.toFixed(1)}ms</span>
              <span className="metric-label">Sparse</span>
            </div>
          )}

          {data.latency_metrics?.rrf_fusion_ms !== undefined && (
            <div className="metric-card">
              <span className="metric-value">{data.latency_metrics.rrf_fusion_ms.toFixed(1)}ms</span>
              <span className="metric-label">RRF</span>
            </div>
          )}
          
          {data.latency_metrics?.generation_ms !== undefined && (
            <div className="metric-card">
              <span className="metric-value">{data.latency_metrics.generation_ms.toFixed(1)}ms</span>
              <span className="metric-label">LLM Gen</span>
            </div>
          )}
          
          {data.latency_metrics?.grounding_ms !== undefined && (
            <div className="metric-card">
              <span className="metric-value">{data.latency_metrics.grounding_ms.toFixed(1)}ms</span>
              <span className="metric-label">Grounding</span>
            </div>
          )}
        </div>
        
        {/* Routing & Guardrails Display */}
        <div className="metrics-grid" style={{ marginTop: '12px', paddingTop: '12px' }}>
          {data.routing && (
            <div className="metric-card">
              <span className="metric-value" style={{ fontSize: '0.9rem', color: 'var(--text-primary)' }}>
                {data.routing.language}
              </span>
              <span className="metric-label">Detected Lang</span>
            </div>
          )}
          
          {data.routing && (
            <div className="metric-card">
              <span className={`intent-badge ${data.routing.intent}`} style={{ fontSize: '0.65rem', padding: '3px 8px' }}>
                {INTENT_ICONS[data.routing.intent] || ""} {data.routing.intent}
              </span>
              <span className="metric-label">Intent</span>
            </div>
          )}

          {data.retrieval_confidence && (
            <div className="metric-card">
              <span className={`badge ${data.retrieval_confidence}`}>
                {data.retrieval_confidence}
              </span>
              <span className="metric-label">Confidence</span>
            </div>
          )}

          {data.grounding && (
            <div className="metric-card">
              <span className={`badge ${data.grounding.status}`}>
                {data.grounding.status}
              </span>
              <span className="metric-label">Grounding</span>
            </div>
          )}
        </div>
        
        {/* Source Citations */}
        {data.sources && data.sources.length > 0 && (
          <div style={{ marginTop: '20px', borderTop: '1px solid var(--surface-border)', paddingTop: '16px' }}>
            <h4 style={{ fontSize: '0.85rem', color: 'var(--text-secondary)', marginBottom: '8px', textTransform: 'uppercase' }}>
              Sources Retrieved
            </h4>
            <div style={{ display: 'flex', gap: '8px', flexWrap: 'wrap' }}>
              {data.sources.slice(0, 3).map((src, i) => (
                <div key={i} style={{ 
                  background: 'rgba(255,255,255,0.05)', 
                  padding: '6px 12px', 
                  borderRadius: '16px',
                  fontSize: '0.8rem',
                  display: 'flex',
                  alignItems: 'center',
                  gap: '6px'
                }}>
                  <span style={{ color: 'var(--accent-color)' }}>★ {src.relevance}</span>
                  <span>{src.strategy}</span>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    );
  };

  return (
    <main className="app-container">
      {/* Top Navigation */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '24px' }}>
        <Link 
          href="/" 
          style={{
            display: 'inline-flex',
            alignItems: 'center',
            gap: '8px',
            color: '#FFE600',
            textDecoration: 'none',
            fontSize: '0.9rem',
            fontWeight: 600,
            background: 'rgba(6, 26, 18, 0.8)',
            border: '1px solid rgba(255, 230, 0, 0.4)',
            padding: '8px 16px',
            borderRadius: '9999px',
            transition: 'all 0.2s ease',
          }}
          className="back-home-link"
        >
          <span>←</span> Back to HH Goa Home
        </Link>
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
          <span style={{ 
            background: 'rgba(255, 19, 125, 0.2)', 
            color: '#FF137D', 
            border: '1px solid rgba(255, 19, 125, 0.4)', 
            padding: '4px 12px', 
            borderRadius: '9999px', 
            fontSize: '0.75rem', 
            fontWeight: 700,
            letterSpacing: '0.05em' 
          }}>
            LIVE RAG PIPELINE
          </span>
        </div>
      </div>

      <div className="header">
        <h1 className="gradient-text">HH Goa 2026</h1>
        <p>Voice-Enabled Multilingual RAG</p>
      </div>
      
      {/* Benchmark Banner */}
      <div className="benchmark-banner">
        <div>
          <h3 style={{ fontSize: '1.1rem', marginBottom: '4px' }}>Production Pipeline Evaluated</h3>
          <p style={{ fontSize: '0.85rem', color: 'var(--text-secondary)' }}>
            Verified 100-query Hindi IndicMSMARCO evaluation (Benchmark-Subset Results).<br/>
            Configuration: <strong>Hybrid RRF (BGE-M3 + BM25)</strong>
          </p>
        </div>
        <div className="benchmark-stats">
          <div className="bench-stat">
            <span className="bench-val">80.0%</span>
            <span className="bench-label">Recall@10</span>
          </div>
          <div className="bench-stat">
            <span className="bench-val">50.4ms</span>
            <span className="bench-label">Retrieval P50</span>
          </div>
          <div className="bench-stat">
            <span className="bench-val" style={{ color: 'var(--warning-color)'}}>54.9ms</span>
            <span className="bench-label">Retrieval P70</span>
          </div>
          <div className="bench-stat">
            <span className="bench-val" style={{ color: 'var(--error-color)'}}>65.4ms</span>
            <span className="bench-label">Retrieval P100</span>
          </div>
        </div>
      </div>

      <div className="glass-container" style={{ minHeight: '60vh', display: 'flex', flexDirection: 'column' }}>
        
        {/* Chat Area */}
        <div className="chat-window" style={{ flexGrow: 1, paddingBottom: '100px' }}>
          {messages.length === 0 ? (
            <div style={{ textAlign: 'center', color: 'var(--text-secondary)', marginTop: '40px' }}>
              <p>Tap the microphone or type a question below.</p>
              <p style={{ fontSize: '0.85rem', marginTop: '8px' }}>E.g., &quot;What is the capital of India?&quot; or &quot;भारत की राजधानी क्या है?&quot;</p>
            </div>
          ) : (
            messages.map((msg, idx) => (
              msg.role === "user" ? (
                <div key={idx} className="message user">
                  {typeof msg.content === 'string' ? msg.content : "..."}
                </div>
              ) : (
                <div key={idx} style={{ width: '100%' }}>
                  {renderAgentMessage(msg.content as RAGResponse)}
                </div>
              )
            ))
          )}
          
          {isProcessing && (
            <div className="message agent" style={{ alignSelf: 'flex-start', padding: '12px 20px', display: 'flex', gap: '8px', alignItems: 'center', width: 'fit-content' }}>
              <div className="spinner" style={{ width: '16px', height: '16px', border: '2px solid var(--accent-color)', borderTopColor: 'transparent', borderRadius: '50%', animation: 'spin 1s linear infinite' }} />
              <span style={{ color: 'var(--text-secondary)', fontSize: '0.9rem' }}>Processing pipeline...</span>
            </div>
          )}
          <div ref={chatEndRef} />
        </div>

        {/* Floating Controls */}
        <div style={{ 
          position: 'fixed', 
          bottom: '30px', 
          left: '50%', 
          transform: 'translateX(-50%)',
          zIndex: 10,
          width: '100%',
          maxWidth: '700px',
          padding: '0 20px',
        }}>
          {/* Mode Switcher */}
          <div className="mode-switcher" style={{ maxWidth: '240px', margin: '0 auto 12px auto' }}>
            <button 
              className={`mode-btn ${inputMode === 'voice' ? 'active' : ''}`} 
              onClick={() => setInputMode('voice')}
            >
              🎤 Voice
            </button>
            <button 
              className={`mode-btn ${inputMode === 'text' ? 'active' : ''}`} 
              onClick={() => setInputMode('text')}
            >
              ⌨️ Text
            </button>
          </div>

          {inputMode === "voice" ? (
            <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
              {/* Strands animation backdrop while recording */}
              <div style={{
                position: 'relative',
                width: '280px',
                height: '120px',
                display: 'flex',
                justifyContent: 'center',
                alignItems: 'center',
              }}>
                <div style={{
                  position: 'absolute',
                  top: '50%',
                  left: '50%',
                  transform: 'translate(-50%, -50%)',
                  width: '280px',
                  height: '280px',
                  borderRadius: '50%',
                  overflow: 'hidden',
                  opacity: isRecording ? 1 : 0,
                  transition: 'opacity 0.4s ease',
                  pointerEvents: 'none',
                  zIndex: 0
                }}>
                  {isRecording && (
                    <Strands
                      colors={["#8B5CF6", "#3B82F6", "#06B6D4"]}
                      count={4}
                      speed={0.8}
                      amplitude={1.2}
                      waviness={1.2}
                      thickness={0.8}
                      glow={3.0}
                      taper={2}
                      spread={1.2}
                      intensity={0.8}
                      saturation={1.8}
                      opacity={1}
                      scale={1.5}
                      glass={false}
                      style={{}}
                    />
                  )}
                </div>

                <div className={`mic-btn-wrapper ${isRecording ? 'recording' : ''}`} style={{ position: 'relative', zIndex: 2, margin: 0 }}>
                  {isRecording && <div className="ripple"></div>}
                  <button 
                    className={`mic-btn ${isRecording ? 'recording' : ''}`}
                    onMouseDown={startRecording}
                    onMouseUp={stopRecording}
                    onTouchStart={startRecording}
                    onTouchEnd={stopRecording}
                    disabled={isProcessing}
                  >
                    <svg viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                      <path d="M12 14c1.66 0 3-1.34 3-3V5c0-1.66-1.34-3-3-3S9 3.34 9 5v6c0 1.66 1.34 3 3 3z"/>
                      <path d="M17 11c0 2.76-2.24 5-5 5s-5-2.24-5-5H5c0 3.53 2.61 6.43 6 6.92V21h2v-3.08c3.39-.49 6-3.39 6-6.92h-2z"/>
                    </svg>
                  </button>
                </div>
              </div>
              <div style={{ textAlign: 'center', fontSize: '0.8rem', color: 'var(--text-secondary)', marginTop: '4px' }}>
                {isRecording ? "Release to Send" : "Hold to Speak"}
              </div>
            </div>
          ) : (
            <div className="text-input-container">
              <input 
                className="text-input"
                type="text"
                placeholder="Type your question... (Hindi, Marathi, or English)"
                value={textQuery}
                onChange={(e) => setTextQuery(e.target.value)}
                onKeyDown={handleKeyDown}
                disabled={isProcessing}
                id="text-query-input"
              />
              <button 
                className="send-btn" 
                onClick={sendTextQuery}
                disabled={isProcessing || !textQuery.trim()}
                id="send-query-btn"
              >
                <svg viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                  <path d="M2.01 21L23 12 2.01 3 2 10l15 2-15 2z"/>
                </svg>
              </button>
            </div>
          )}
        </div>

      </div>

      {/* Toast Notifications */}
      {toasts.length > 0 && (
        <div className="toast-container">
          {toasts.map(toast => (
            <div key={toast.id} className={`toast ${toast.type} ${toast.dismissing ? 'dismissing' : ''}`}>
              <span className="toast-icon">{toastIcons[toast.type]}</span>
              <span className="toast-message">{toast.message}</span>
            </div>
          ))}
        </div>
      )}
      
      <style dangerouslySetInnerHTML={{__html: `
        @keyframes spin { 100% { transform: rotate(360deg); } }
        .back-home-link:hover {
          background: rgba(255, 230, 0, 0.15) !important;
          border-color: #FFE600 !important;
          transform: translateX(-3px);
        }
      `}} />
    </main>
  );
}
