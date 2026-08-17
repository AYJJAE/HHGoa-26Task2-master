import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from api.sarvam_client import SarvamClient
from pipeline.query_router import QueryRouter


class FakeResponse:
    status_code = 200

    def __init__(self, payload):
        self.payload = payload

    def json(self):
        return self.payload


def test_multilingual_response_preserves_unicode_and_detected_language(monkeypatch):
    monkeypatch.setenv("SARVAM_API_KEY", "test-key")
    client = SarvamClient()
    responses = [
        {"transcript": "What documents are required?", "language_code": "en-IN"},
        {"transcript": "मुझे कौन से दस्तावेज़ चाहिए?", "language_code": "hi-IN"},
        {"transcript": "मला कोणती कागदपत्रे लागतील?", "language_code": "mr-IN"},
        {"transcript": "म्हाका कोणती कागदपत्रां जाय?", "language_code": "kok-IN"},
    ]
    monkeypatch.setattr(client.session, "post", lambda *args, **kwargs: FakeResponse(responses.pop(0)))

    values = [client.transcribe_audio(b"audio", "voice.webm", "audio/webm") for _ in range(4)]
    assert [item.language for item in values] == ["en-IN", "hi-IN", "mr-IN", "kok-IN"]
    assert values[1].text == "मुझे कौन से दस्तावेज़ चाहिए?"
    assert values[2].text == "मला कोणती कागदपत्रे लागतील?"
    assert values[3].text == "म्हाका कोणती कागदपत्रां जाय?"


def test_auto_detection_uses_saaras_codemix_and_keeps_mixed_transcript(monkeypatch):
    monkeypatch.setenv("SARVAM_API_KEY", "test-key")
    client = SarvamClient()
    captured = {}
    def post(*args, **kwargs):
        captured.update(kwargs)
        return FakeResponse({"transcript": "मला tomorrow office ला जायचं आहे?", "language_code": "mr-IN"})
    monkeypatch.setattr(client.session, "post", post)

    result = client.transcribe_audio(b"audio", "voice.webm", "audio/webm", "auto")
    assert result.success and result.text == "मला tomorrow office ला जायचं आहे?"
    assert captured["data"] == {"model": "saaras:v3", "mode": "codemix", "language_code": "unknown"}


def test_transcription_errors_are_safe_and_structured(monkeypatch):
    monkeypatch.setenv("SARVAM_API_KEY", "test-key")
    client = SarvamClient()
    assert client.transcribe_audio(b"").error == "No audio was provided."
    assert client.transcribe_audio(b"audio", "voice.txt", "text/plain").error == "Unsupported audio format."


def test_rag_router_preserves_the_speech_language_metadata():
    routing = QueryRouter().route_query("मला tomorrow office ला जायचं आहे?", language_hint="mr-IN")
    assert routing["language"] == "Marathi"
