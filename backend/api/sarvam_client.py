"""Sarvam STT adapter. It transcribes speech; it never translates it."""
import io
import mimetypes
import os
import re
import time
from dataclasses import dataclass
from typing import Optional
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

LANGUAGE_HINTS = {
    "auto": "unknown", "unknown": "unknown",
    "en": "en-IN", "en-in": "en-IN", "english": "en-IN",
    "hi": "hi-IN", "hi-in": "hi-IN", "hindi": "hi-IN",
    "mr": "mr-IN", "mr-in": "mr-IN", "marathi": "mr-IN",
    "kok": "kok-IN", "kok-in": "kok-IN", "konkani": "kok-IN",
}
ALLOWED_CONTENT_TYPES = {
    "audio/wav", "audio/x-wav", "audio/mpeg", "audio/mp3", "audio/aac",
    "audio/ogg", "audio/opus", "audio/flac", "audio/webm", "audio/mp4",
    "video/webm", "video/mp4", "application/octet-stream",
}


@dataclass(frozen=True)
class TranscriptionResult:
    success: bool
    text: str = ""
    language: Optional[str] = None
    language_probability: Optional[float] = None
    error: Optional[str] = None
    provider: Optional[str] = None
    fallback_used: bool = False
    latency_ms: float = 0.0


class SarvamClient:
    def __init__(self):
        self._initial_api_key = (
            os.environ.get("SARVAM_API_KEY")
            or os.environ.get("SARVAM_AI_API_KEY")
            or ""
        ).strip().strip("\"'")
        self.base_url = os.environ.get("SARVAM_STT_BASE_URL", "https://api.sarvam.ai").rstrip("/")
        self.model = os.environ.get("SARVAM_STT_MODEL", "saaras:v3")
        self.mode = os.environ.get("SARVAM_STT_MODE", "codemix")
        self.timeout_seconds = float(os.environ.get("SARVAM_STT_TIMEOUT_SECONDS", "20"))
        self.max_audio_bytes = int(os.environ.get("SARVAM_STT_MAX_AUDIO_BYTES", str(15 * 1024 * 1024)))
        
        # Persistent session with connection pooling
        self.session = requests.Session()
        retry_strategy = Retry(
            total=2,
            backoff_factor=0.5,
            status_forcelist=[500, 502, 503, 504],
            allowed_methods=["POST"]
        )
        adapter = HTTPAdapter(max_retries=retry_strategy, pool_connections=10, pool_maxsize=10)
        self.session.mount("https://", adapter)
        self.session.mount("http://", adapter)

    @property
    def api_key(self) -> str:
        return (
            os.environ.get("SARVAM_API_KEY")
            or os.environ.get("SARVAM_AI_API_KEY")
            or self._initial_api_key
            or ""
        ).strip().strip("\"'")

    @api_key.setter
    def api_key(self, value: str):
        self._initial_api_key = value

    @staticmethod
    def _clean_text(text: object) -> str:
        return re.sub(r"\s+", " ", str(text or "")).strip()

    @staticmethod
    def normalize_language_hint(language_hint: Optional[str]) -> str:
        return LANGUAGE_HINTS.get((language_hint or "auto").strip().casefold(), "unknown")

    def transcribe_audio(
        self,
        audio_bytes: bytes,
        filename: str = "audio.webm",
        content_type: Optional[str] = None,
        language_hint: Optional[str] = None,
    ) -> TranscriptionResult:
        """Call Sarvam Saaras v3 once with auto-detection or a validated hint."""
        t0 = time.perf_counter()
        
        if not audio_bytes:
            return TranscriptionResult(False, error="No audio was provided.", provider="sarvam", fallback_used=True, latency_ms=0.0)
        if len(audio_bytes) > self.max_audio_bytes:
            return TranscriptionResult(False, error="Audio file is too large. Please submit a shorter recording.", provider="sarvam", fallback_used=True, latency_ms=0.0)
            
        key = self.api_key
        if not key:
            print("[STT:Sarvam] Sarvam configured: false")
            return TranscriptionResult(False, error="Voice transcription is not configured for Sarvam.", provider="sarvam", fallback_used=True, latency_ms=0.0)

        mime_type = content_type or mimetypes.guess_type(filename or "")[0] or "application/octet-stream"
        base_mime = mime_type.split(";")[0].strip().lower()
        if base_mime not in ALLOWED_CONTENT_TYPES and mime_type not in ALLOWED_CONTENT_TYPES:
            return TranscriptionResult(False, error="Unsupported audio format.", provider="sarvam", fallback_used=True, latency_ms=0.0)

        clean_filename = filename or "recording.webm"

        data = {
            "model": self.model,
            "mode": self.mode,
            "language_code": self.normalize_language_hint(language_hint),
        }
        files = {"file": (clean_filename, io.BytesIO(audio_bytes), base_mime)}
        headers = {"api-subscription-key": key}
        
        try:
            response = self.session.post(
                f"{self.base_url}/speech-to-text",
                headers=headers,
                files=files,
                data=data,
                timeout=(3.0, self.timeout_seconds),
            )
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            
            if response.status_code == 200:
                result = response.json()
                text = self._clean_text(result.get("transcript"))
                if not text:
                    print(f"[STT:Sarvam] Succeeded in {elapsed_ms:.1f}ms but no speech text returned.")
                    return TranscriptionResult(False, error="No speech was detected.", provider="sarvam", fallback_used=True, latency_ms=elapsed_ms)
                probability = result.get("language_probability")
                try:
                    probability = float(probability) if probability is not None else None
                except (TypeError, ValueError):
                    probability = None
                detected_lang = result.get("language_code")
                safe_snippet = text[:40].encode("ascii", "backslashreplace").decode("ascii")
                print(f"[STT:Sarvam] Fallback transcription succeeded in {elapsed_ms:.1f}ms (lang={detected_lang}): '{safe_snippet}...'")
                return TranscriptionResult(
                    True,
                    text=text,
                    language=detected_lang,
                    language_probability=probability,
                    provider="sarvam",
                    fallback_used=True,
                    latency_ms=elapsed_ms,
                )
            
            if response.status_code == 429:
                message = "The transcription service is busy. Please try again shortly."
            elif response.status_code in (400, 422):
                message = "The audio could not be processed. Please use a short recording with audible speech."
            elif response.status_code in (401, 403):
                message = "Sarvam API key is invalid or unauthorized."
            else:
                message = f"Sarvam API returned HTTP {response.status_code}"
                
            print(f"[STT:Sarvam] HTTP {response.status_code} in {elapsed_ms:.1f}ms. Message: {message}")
            return TranscriptionResult(False, error=message, provider="sarvam", fallback_used=True, latency_ms=elapsed_ms)
            
        except requests.Timeout:
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            print(f"[STT:Sarvam] Timeout after {elapsed_ms:.1f}ms")
            return TranscriptionResult(False, error="Transcription timed out. Please try a shorter recording.", provider="sarvam", fallback_used=True, latency_ms=elapsed_ms)
        except Exception as exc:
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            print(f"[STT:Sarvam] Request failed in {elapsed_ms:.1f}ms: {type(exc).__name__}")
            return TranscriptionResult(False, error=f"Sarvam request failed: {type(exc).__name__}", provider="sarvam", fallback_used=True, latency_ms=elapsed_ms)
