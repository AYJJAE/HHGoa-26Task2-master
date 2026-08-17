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
        # Default 5000ms timeout for primary failover if not specified
        self.primary_timeout_ms = int(os.environ.get("ELEVENLABS_TIMEOUT_MS", "5000"))

    def transcribe_audio(
        self,
        audio_bytes: bytes,
        filename: str = "audio.webm",
        content_type: Optional[str] = None,
        language_hint: Optional[str] = None,
    ) -> TranscriptionResult:
        
        start_time = time.time()
        
        # Override primary client's internal timeout to enforce the failover SLA
        original_timeout = self.primary.timeout_seconds
        self.primary.timeout_seconds = self.primary_timeout_ms / 1000.0
        
        try:
            result = self.primary.transcribe_audio(audio_bytes, filename, content_type, language_hint)
        except Exception as e:
            print(f"Primary STT (ElevenLabs) threw an unhandled exception: {e}")
            result = TranscriptionResult(success=False, error=str(e))
        finally:
            self.primary.timeout_seconds = original_timeout

        latency_ms = (time.time() - start_time) * 1000.0

        if result.success and result.text:
            return TranscriptionResult(
                success=True,
                text=result.text,
                language=result.language,
                language_probability=result.language_probability,
                error=None,
                provider="elevenlabs",
                fallback_used=False,
                latency_ms=latency_ms
            )
            
        print(f"Primary STT (ElevenLabs) failed with error: {result.error}. Triggering Sarvam fallback.")
        
        # --- Fallback Path ---
        fallback_start_time = time.time()
        try:
            fallback_result = self.fallback.transcribe_audio(audio_bytes, filename, content_type, language_hint)
        except Exception as e:
            print(f"Fallback STT (Sarvam) threw an unhandled exception: {e}")
            fallback_result = TranscriptionResult(success=False, error=str(e))
            
        fallback_latency_ms = (time.time() - fallback_start_time) * 1000.0
        total_latency_ms = (time.time() - start_time) * 1000.0
        
        if fallback_result.success and fallback_result.text:
            return TranscriptionResult(
                success=True,
                text=fallback_result.text,
                language=fallback_result.language,
                language_probability=fallback_result.language_probability,
                error=None,
                provider="sarvam",
                fallback_used=True,
                latency_ms=total_latency_ms # We return the total latency perceived by the user
            )
            
        # If both fail, return a clean error
        return TranscriptionResult(
            success=False,
            error="Speech transcription unavailable.",
            provider=None,
            fallback_used=True,
            latency_ms=total_latency_ms
        )
