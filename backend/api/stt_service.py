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
        Executes primary transcription with ElevenLabs Scribe.
        Fails over to Sarvam Saaras v3 immediately if ElevenLabs fails or is not configured.
        """
        total_t0 = time.perf_counter()
        
        el_key_present = bool(self.primary.api_key)
        sv_key_present = bool(self.fallback.api_key)
        print(f"[STT] ElevenLabs configured: {el_key_present}")
        print(f"[STT] Sarvam configured: {sv_key_present}")
        
        # 1. PRIMARY PROVIDER: ElevenLabs
        print("[STT] attempting ElevenLabs")
        original_timeout = self.primary.timeout_seconds
        self.primary.timeout_seconds = self.primary_timeout_ms / 1000.0
        
        try:
            print("[STT] ELEVENLABS REQUEST START")
            primary_res = self.primary.transcribe_audio(
                audio_bytes=audio_bytes,
                filename=filename,
                content_type=content_type,
                language_hint=language_hint
            )
            print("[STT] ELEVENLABS REQUEST END")
        except Exception as e:
            import traceback
            print(f"[STT] ELEVENLABS EXCEPTION: {type(e).__name__} - {e}\n{traceback.format_exc()}")
            safe_msg = str(e) if str(e) else type(e).__name__
            primary_res = TranscriptionResult(success=False, error=safe_msg, provider="elevenlabs")
        finally:
            self.primary.timeout_seconds = original_timeout

        if primary_res.success and primary_res.text:
            total_latency = (time.perf_counter() - total_t0) * 1000.0
            print(f"[STT] ELEVENLABS SUCCESS | latency={primary_res.latency_ms:.1f}ms (total={total_latency:.1f}ms)")
            return TranscriptionResult(
                success=True,
                text=primary_res.text,
                language=primary_res.language,
                language_probability=primary_res.language_probability,
                error=None,
                provider="elevenlabs",
                fallback_used=False,
                latency_ms=primary_res.latency_ms,
            )
            
        print(f"[STT] ELEVENLABS FAILED | error_type='{primary_res.error}' | latency={primary_res.latency_ms:.1f}ms")
        print("[STT] falling back to SARVAM")
        
        # 2. FALLBACK PROVIDER: Sarvam
        try:
            print("[STT] SARVAM REQUEST START")
            fallback_res = self.fallback.transcribe_audio(
                audio_bytes=audio_bytes,
                filename=filename,
                content_type=content_type,
                language_hint=language_hint
            )
            print("[STT] SARVAM REQUEST END")
        except Exception as e:
            import traceback
            print(f"[STT] SARVAM EXCEPTION: {type(e).__name__} - {e}\n{traceback.format_exc()}")
            safe_msg = str(e) if str(e) else type(e).__name__
            fallback_res = TranscriptionResult(success=False, error=safe_msg, provider="sarvam", fallback_used=True)
            
        total_latency = (time.perf_counter() - total_t0) * 1000.0
        
        if fallback_res.success and fallback_res.text:
            print(f"[STT] SARVAM SUCCESS | latency={fallback_res.latency_ms:.1f}ms (total={total_latency:.1f}ms)")
            return TranscriptionResult(
                success=True,
                text=fallback_res.text,
                language=fallback_res.language,
                language_probability=fallback_res.language_probability,
                error=None,
                provider="sarvam",
                fallback_used=True,
                latency_ms=fallback_res.latency_ms,
            )
            
        # 3. BOTH PROVIDERS FAILED
        print(f"[STT] SARVAM FAILED | error_type='{fallback_res.error}' | latency={fallback_res.latency_ms:.1f}ms")
        print(f"[STT] Both ElevenLabs and Sarvam failed. Total STT elapsed: {total_latency:.1f}ms.")
        return TranscriptionResult(
            success=False,
            error=fallback_res.error or primary_res.error or "Speech transcription unavailable.",
            provider=None,
            fallback_used=True,
            latency_ms=total_latency,
        )
