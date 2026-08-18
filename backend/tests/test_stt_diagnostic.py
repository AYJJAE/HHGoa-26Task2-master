import os
import sys
import asyncio
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

from api.stt_service import STTService

async def main():
    print("=== STT DIAGNOSTIC ===")
    print(f"ELEVENLABS_API_KEY Configured: {'ELEVENLABS_API_KEY' in os.environ}")
    print(f"SARVAM_API_KEY Configured: {'SARVAM_API_KEY' in os.environ}")
    
    # We use a real audio file for testing
    audio_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), "test.wav")
    if not os.path.exists(audio_file):
        print(f"ERROR: No test audio file found at {audio_file}")
        return
        
    with open(audio_file, "rb") as f:
        audio_bytes = f.read()
        
    print(f"Loaded test audio: {len(audio_bytes)} bytes")
    
    stt = STTService()
    
    print("\n--- Testing Primary STT (ElevenLabs) ---")
    t0 = time.perf_counter()
    res = await asyncio.to_thread(stt.primary.transcribe_audio, audio_bytes, "test.wav", "audio/wav", "en")
    t1 = time.perf_counter()
    print(f"Result: success={res.success}, text={res.text}, error={res.error}, latency={res.latency_ms:.2f}ms")
    print(f"Elapsed test script time: {(t1-t0)*1000:.2f}ms")
    
    print("\n--- Testing Fallback STT (Sarvam) ---")
    t0 = time.perf_counter()
    res = await asyncio.to_thread(stt.fallback.transcribe_audio, audio_bytes, "test.wav", "audio/wav", "en")
    t1 = time.perf_counter()
    print(f"Result: success={res.success}, text={res.text}, error={res.error}, latency={res.latency_ms:.2f}ms")
    print(f"Elapsed test script time: {(t1-t0)*1000:.2f}ms")

def test_diagnostic():
    asyncio.run(main())

if __name__ == "__main__":
    test_diagnostic()
