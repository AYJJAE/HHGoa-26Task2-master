"""ElevenLabs STT adapter. It transcribes speech using Scribe v2."""
import io
import mimetypes
import os
import re
import time
from typing import Optional
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from .sarvam_client import TranscriptionResult, ALLOWED_CONTENT_TYPES


class ElevenLabsClient:
    def __init__(self):
        self._initial_api_key = (
            os.environ.get("ELEVENLABS_API_KEY")
            or os.environ.get("ELEVEN_LABS_API_KEY")
            or os.environ.get("XI_API_KEY")
            or ""
        ).strip().strip("\"'")
        self.base_url = os.environ.get("ELEVENLABS_STT_BASE_URL", "https://api.elevenlabs.io/v1").rstrip("/")
        self.model = os.environ.get("ELEVENLABS_STT_MODEL", "scribe_v1")
        self.timeout_seconds = float(os.environ.get("ELEVENLABS_STT_TIMEOUT_SECONDS", "15"))
        self.max_audio_bytes = int(os.environ.get("ELEVENLABS_STT_MAX_AUDIO_BYTES", str(25 * 1024 * 1024)))
        
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
            os.environ.get("ELEVENLABS_API_KEY")
            or os.environ.get("ELEVEN_LABS_API_KEY")
            or os.environ.get("XI_API_KEY")
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
    def _normalize_mime_type(mime: str) -> str:
        """Strip codecs parameters from mime types (e.g. 'audio/webm;codecs=opus' -> 'audio/webm')."""
        if not mime:
            return "audio/webm"
        base = mime.split(";")[0].strip().lower()
        if base in ("video/webm", "application/octet-stream"):
            return "audio/webm"
        return base

    def transcribe_audio(
        self,
        audio_bytes: bytes,
        filename: str = "audio.webm",
        content_type: Optional[str] = None,
        language_hint: Optional[str] = None,
    ) -> TranscriptionResult:
        """Call ElevenLabs Scribe v1/v2 for transcription."""
        t0 = time.perf_counter()
        
        if not audio_bytes:
            return TranscriptionResult(False, error="No audio was provided.", provider="elevenlabs", latency_ms=0.0)
        if len(audio_bytes) > self.max_audio_bytes:
            return TranscriptionResult(False, error="Audio file is too large. Please submit a shorter recording.", provider="elevenlabs", latency_ms=0.0)
            
        key = self.api_key
        if not key:
            print("[STT:ElevenLabs] ElevenLabs configured: false")
            return TranscriptionResult(False, error="Voice transcription is not configured for ElevenLabs.", provider="elevenlabs", latency_ms=0.0)

        raw_mime = content_type or mimetypes.guess_type(filename or "")[0] or "audio/webm"
        mime_type = self._normalize_mime_type(raw_mime)
        clean_filename = filename or "recording.webm"
        if not any(clean_filename.endswith(ext) for ext in (".webm", ".wav", ".mp3", ".ogg", ".m4a", ".mp4", ".flac", ".aac")):
            clean_filename = "recording.webm"

        data = {
            "model_id": self.model,
        }
        
        if language_hint and language_hint.lower() not in ("auto", "unknown"):
            lang = language_hint.split('-')[0].lower()
            data["language_code"] = lang

        files = {"file": (clean_filename, io.BytesIO(audio_bytes), mime_type)}
        headers = {"xi-api-key": key}
        
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
                text = self._clean_text(result.get("text"))
                if not text:
                    print(f"[STT:ElevenLabs] Succeeded in {elapsed_ms:.1f}ms but no speech text returned.")
                    return TranscriptionResult(False, error="No speech was detected.", provider="elevenlabs", latency_ms=elapsed_ms)
                detected_lang = result.get("language_code") or result.get("language")
                safe_snippet = text[:40].encode("ascii", "backslashreplace").decode("ascii")
                print(f"[STT:ElevenLabs] Transcription succeeded in {elapsed_ms:.1f}ms (lang={detected_lang}): '{safe_snippet}...'")
                return TranscriptionResult(
                    True,
                    text=text,
                    language=detected_lang,
                    language_probability=result.get("language_probability"),
                    provider="elevenlabs",
                    latency_ms=elapsed_ms,
                )
            
            if response.status_code == 429:
                message = "The transcription service is busy. Please try again shortly."
            elif response.status_code in (400, 422):
                message = "The audio could not be processed."
            elif response.status_code in (401, 403):
                message = "ElevenLabs API key is invalid or unauthorized."
            else:
                message = f"ElevenLabs API returned HTTP {response.status_code}"
            
            print(f"[STT:ElevenLabs] HTTP {response.status_code} in {elapsed_ms:.1f}ms. Message: {message}")
            return TranscriptionResult(False, error=message, provider="elevenlabs", latency_ms=elapsed_ms)
            
        except requests.Timeout:
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            print(f"[STT:ElevenLabs] Timeout after {elapsed_ms:.1f}ms")
            return TranscriptionResult(False, error="Transcription timed out. Please try a shorter recording.", provider="elevenlabs", latency_ms=elapsed_ms)
        except Exception as exc:
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            print(f"[STT:ElevenLabs] Request failed in {elapsed_ms:.1f}ms: {type(exc).__name__}")
            return TranscriptionResult(False, error=f"ElevenLabs request failed: {type(exc).__name__}", provider="elevenlabs", latency_ms=elapsed_ms)
