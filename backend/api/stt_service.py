"""Unified STT Service with Failover Architecture."""
import time
import os
from typing import Optional
from .sarvam_client import SarvamClient, TranscriptionResult
from .elevenlabs_client import ElevenLabsClient


class STTService:
    def __init__(self):
        self.primary = ElevenLabsClient()
        self.fallback = SarvamClient()
        # Default 6000ms timeout for primary failover if not specified
        self.primary_timeout_ms = int(os.environ.get("ELEVENLABS_TIMEOUT_MS", "6000"))

    def transcribe_audio(
        self,
        audio_bytes: bytes,
        filename: str = "audio.webm",
        content_type: Optional[str] = None,
        language_hint: Optional[str] = None,
    ) -> TranscriptionResult:
        """
        Executes primary transcription with ElevenLabs Scribe v2.
        Fails over to Sarvam Saaras v3 immediately if ElevenLabs fails or is not configured.
        """
        total_t0 = time.perf_counter()
        
        # 1. PRIMARY PROVIDER: ElevenLabs
        print("[STT Pipeline] Attempting Primary STT: ElevenLabs Scribe v2...")
        original_timeout = self.primary.timeout_seconds
        self.primary.timeout_seconds = self.primary_timeout_ms / 1000.0
        
        try:
            primary_res = self.primary.transcribe_audio(
                audio_bytes=audio_bytes,
                filename=filename,
                content_type=content_type,
                language_hint=language_hint
            )
        except Exception as e:
            print(f"[STT Pipeline] Primary STT (ElevenLabs) unhandled error: {type(e).__name__}")
            primary_res = TranscriptionResult(success=False, error=str(e), provider="elevenlabs")
        finally:
            self.primary.timeout_seconds = original_timeout

        if primary_res.success and primary_res.text:
            total_latency = (time.perf_counter() - total_t0) * 1000.0
            print(f"[STT Pipeline] ElevenLabs succeeded in {primary_res.latency_ms:.1f}ms (total STT: {total_latency:.1f}ms).")
            return TranscriptionResult(
                success=True,
                text=primary_res.text,
                language=primary_res.language,
                language_probability=primary_res.language_probability,
                error=None,
                provider="elevenlabs",
                fallback_used=False,
                latency_ms=total_latency,
            )
            
        print(f"[STT Pipeline] Primary STT (ElevenLabs) failed: '{primary_res.error}'. Attempting Fallback STT: Sarvam Saaras v3...")
        
        # 2. FALLBACK PROVIDER: Sarvam
        try:
            fallback_res = self.fallback.transcribe_audio(
                audio_bytes=audio_bytes,
                filename=filename,
                content_type=content_type,
                language_hint=language_hint
            )
        except Exception as e:
            print(f"[STT Pipeline] Fallback STT (Sarvam) unhandled error: {type(e).__name__}")
            fallback_res = TranscriptionResult(success=False, error=str(e), provider="sarvam", fallback_used=True)
            
        total_latency = (time.perf_counter() - total_t0) * 1000.0
        
        if fallback_res.success and fallback_res.text:
            print(f"[STT Pipeline] Sarvam fallback succeeded in {fallback_res.latency_ms:.1f}ms (total STT: {total_latency:.1f}ms).")
            return TranscriptionResult(
                success=True,
                text=fallback_res.text,
                language=fallback_res.language,
                language_probability=fallback_res.language_probability,
                error=None,
                provider="sarvam",
                fallback_used=True,
                latency_ms=total_latency,
            )
            
        # 3. BOTH PROVIDERS FAILED
        print(f"[STT Pipeline] Both ElevenLabs and Sarvam failed. Total STT latency: {total_latency:.1f}ms.")
        return TranscriptionResult(
            success=False,
            error=fallback_res.error or primary_res.error or "Speech transcription unavailable.",
            provider=None,
            fallback_used=True,
            latency_ms=total_latency,
        )
