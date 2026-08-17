"""Sarvam STT adapter. It transcribes speech; it never translates it."""
import io
import mimetypes
import os
import re
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
        self.api_key = os.environ.get("SARVAM_API_KEY", "")
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
        
        if not self.api_key:
            print("WARNING: SARVAM_API_KEY not set. Voice transcription fallback will be unavailable.")

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
        if not audio_bytes:
            return TranscriptionResult(False, error="No audio was provided.")
        if len(audio_bytes) > self.max_audio_bytes:
            return TranscriptionResult(False, error="Audio file is too large. Please submit a shorter recording.")
        if not self.api_key:
            return TranscriptionResult(False, error="Voice transcription is not configured for Sarvam.")

        mime_type = content_type or mimetypes.guess_type(filename)[0] or "application/octet-stream"
        base_mime = mime_type.split(";")[0].strip().lower()
        if base_mime not in ALLOWED_CONTENT_TYPES and mime_type not in ALLOWED_CONTENT_TYPES:
            return TranscriptionResult(False, error="Unsupported audio format.")

        data = {
            "model": self.model,
            "mode": self.mode,
            "language_code": self.normalize_language_hint(language_hint),
        }
        files = {"file": (filename or "audio.webm", io.BytesIO(audio_bytes), base_mime)}
        headers = {"api-subscription-key": self.api_key}
        
        try:
            response = self.session.post(
                f"{self.base_url}/speech-to-text",
                headers=headers,
                files=files,
                data=data,
                timeout=(3.0, self.timeout_seconds),
            )
            
            if response.status_code == 200:
                result = response.json()
                text = self._clean_text(result.get("transcript"))
                if not text:
                    return TranscriptionResult(False, error="No speech was detected.")
                probability = result.get("language_probability")
                try:
                    probability = float(probability) if probability is not None else None
                except (TypeError, ValueError):
                    probability = None
                return TranscriptionResult(
                    True,
                    text=text,
                    language=result.get("language_code"),
                    language_probability=probability,
                )
            
            if response.status_code == 429:
                message = "The transcription service is busy. Please try again shortly."
            elif response.status_code in (400, 422):
                message = "The audio could not be processed. Please use a short recording with audible speech."
            else:
                message = f"Sarvam API returned HTTP {response.status_code}"
                
            return TranscriptionResult(False, error=message)
            
        except requests.Timeout:
            return TranscriptionResult(False, error="Transcription timed out. Please try a shorter recording.")
        except Exception as exc:
            return TranscriptionResult(False, error=f"Sarvam request failed: {type(exc).__name__}")
