import asyncio
import json
import time
import requests
import statistics
import os
from pathlib import Path

def summary(values):
    if not values:
        return {"count": 0, "p50": 0.0, "p70": 0.0, "p100": 0.0}
    values = sorted(values)
    n = len(values)
    return {
        "count": n,
        "p50": round(values[int(0.50 * (n - 1))], 2),
        "p70": round(values[int(0.70 * (n - 1))], 2),
        "p100": round(values[-1], 2),
    }

def run_test(url, audio_path, language="auto"):
    with open(audio_path, 'rb') as f:
        files = {'audio': ('test.wav', f, 'audio/wav')}
        data = {'language': language}
        start = time.perf_counter()
        resp = requests.post(url, files=files, data=data, timeout=30)
        total_ms = (time.perf_counter() - start) * 1000
    
    return resp.json(), total_ms

def main():
    print("\n==========================================")
    print(" Voice STT & RAG End-to-End Benchmark")
    print("==========================================\n")
    url = "http://localhost:8000/api/voice_ask"
    
    # Locate test audio file
    backend_root = Path(__file__).resolve().parents[2]
    audio_path = backend_root / "test.wav"
    if not audio_path.exists():
        audio_path = backend_root / "api" / "test.wav"
    
    if not audio_path.exists():
        print(f"Error: Could not find test.wav at {audio_path}")
        return

    stt_latencies = []
    rag_latencies = []
    total_latencies = []

    print(f"Running 5 real audio queries using {audio_path.name}...\n")
    for i in range(5):
        try:
            data, t_total = run_test(url, audio_path)
            provider = data.get("transcription", {}).get("provider", "unknown")
            text = data.get("transcription", {}).get("text", "")
            stt_ms = data.get("latency_metrics", {}).get("stt_ms", 0.0)
            rag_ms = data.get("latency_metrics", {}).get("total_e2e_ms", t_total) - stt_ms
            
            stt_latencies.append(stt_ms)
            rag_latencies.append(rag_ms)
            total_latencies.append(t_total)
            
            print(f"Run {i+1}: Provider={provider} | STT={stt_ms:.1f}ms | RAG={rag_ms:.1f}ms | Total={t_total:.1f}ms")
            print(f"       Transcript: \"{text[:60]}\"")
        except Exception as e:
            print(f"Run {i+1} failed: {e}")

    print("\n------------------------------------------")
    print(" Performance Telemetry Summary (Real)")
    print("------------------------------------------")
    print(f"{'Metric':<18}{'P50':>10}{'P70':>10}{'P100':>10}  (ms)")
    
    stt_sum = summary(stt_latencies)
    rag_sum = summary(rag_latencies)
    tot_sum = summary(total_latencies)
    
    print(f"{'STT Latency':<18}{stt_sum['p50']:>10.2f}{stt_sum['p70']:>10.2f}{stt_sum['p100']:>10.2f}")
    print(f"{'RAG Latency':<18}{rag_sum['p50']:>10.2f}{rag_sum['p70']:>10.2f}{rag_sum['p100']:>10.2f}")
    print(f"{'TOTAL E2E':<18}{tot_sum['p50']:>10.2f}{tot_sum['p70']:>10.2f}{tot_sum['p100']:>10.2f}")
    print("==========================================\n")

if __name__ == "__main__":
    main()
