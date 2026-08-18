import os
import io
import json
from fastapi.testclient import TestClient
import sys

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from api.main import app
from api.main import resources

client = TestClient(app)

def run_tests():
    resources.initialize()
    print("=== PRODUCTION READINESS STT TEST ===")
    
    with open(os.path.join(os.path.dirname(__file__), "..", "test.wav"), "rb") as f:
        valid_audio = f.read()
        
    empty_audio = b""
    
    # Store original keys to restore later
    original_el_key = resources.stt_client.primary.api_key
    original_sv_key = resources.stt_client.fallback.api_key
    
    def print_result(case_name, res, expected_provider, expected_fallback):
        print(f"\n--- {case_name} ---")
        if res.status_code != 200:
            print(f"FAILED: HTTP {res.status_code}")
            print(res.text)
            return
            
        data = res.json()
        if "transcription" not in data:
            print("FAILED: No transcription block.")
            print(data)
            return
            
        t = data["transcription"]
        lm = data.get("latency_metrics", {})
        
        print(f"success: {t.get('success')}")
        print(f"error: {t.get('error')}")
        print(f"provider: {t.get('provider')}")
        print(f"fallback_used: {t.get('fallback_used')}")
        safe_text = t.get('text', '').encode('ascii', 'backslashreplace').decode('ascii')
        print(f"transcription text: '{safe_text}'")
        print(f"stt_ms: {lm.get('stt_ms', 0):.2f}")
        print(f"transcription_total_ms: {lm.get('transcription_total_ms', 0):.2f}")
        
        if t.get('provider') != expected_provider:
            print(f"FAILED: EXPECTED PROVIDER {expected_provider}, GOT {t.get('provider')}")
        else:
            print("PASS: Provider match")
            
        if t.get('fallback_used') != expected_fallback:
            print(f"FAILED: EXPECTED FALLBACK {expected_fallback}, GOT {t.get('fallback_used')}")
        else:
            print("PASS: Fallback match")

    try:
        # CASE A: ElevenLabs Returns Valid non-empty transcription
        # Since our test.wav has speech, we just need to ensure ElevenLabs has the correct key and we request 'en'
        resources.stt_client.primary.api_key = original_el_key
        resources.stt_client.fallback.api_key = original_sv_key
        
        # Test all languages
        for lang in ["en", "hi", "mr", "kok"]:
            # Note: ElevenLabs usually auto-detects, but if it doesn't support 'kok' perfectly, it might fail or return something.
            # To strictly get Case A, 'en' is safest. But we will test all.
            print(f"\n>> Testing Language: {lang}")
            res = client.post(
                "/api/voice_ask",
                data={"language": lang},
                files={"audio": ("test.webm", valid_audio, "audio/webm;codecs=opus")}
            )
            # The test.wav contains English, so let's see how the providers handle it.
            # Since ElevenLabs might return empty string for Hindi if it only hears English and is forced to 'hi'.
            # We'll just print the result.
            print_result(f"Language Test ({lang})", res, expected_provider=None, expected_fallback=None)


        # CASE B: ElevenLabs fails with an actual API error
        def mock_error(*args, **kwargs):
            raise Exception("Simulated network failure")
            
        original_el_transcribe = resources.stt_client.primary.transcribe_audio
        resources.stt_client.primary.transcribe_audio = mock_error
        
        res = client.post(
            "/api/voice_ask",
            data={"language": "en"},
            files={"audio": ("test.webm", valid_audio, "audio/webm;codecs=opus")}
        )
        print_result("CASE B: ElevenLabs fails -> Sarvam fallback", res, "sarvam", True)


        # CASE C: ElevenLabs returns HTTP 200 but empty/no-speech transcription
        # We can simulate this by mocking _clean_text to return empty string for this test.
        resources.stt_client.primary.transcribe_audio = original_el_transcribe
        original_clean = resources.stt_client.primary._clean_text
        resources.stt_client.primary._clean_text = lambda x: ""
        res = client.post(
            "/api/voice_ask",
            data={"language": "en"},
            files={"audio": ("test.webm", valid_audio, "audio/webm;codecs=opus")}
        )
        print_result("CASE C: ElevenLabs Empty Text -> Sarvam fallback", res, "sarvam", True)
        resources.stt_client.primary._clean_text = original_clean


        # CASE D: Both providers fail
        original_sv_transcribe = resources.stt_client.fallback.transcribe_audio
        resources.stt_client.primary.transcribe_audio = mock_error
        resources.stt_client.fallback.transcribe_audio = mock_error
        res = client.post(
            "/api/voice_ask",
            data={"language": "en"},
            files={"audio": ("test.webm", valid_audio, "audio/webm;codecs=opus")}
        )
        print_result("CASE D: Both providers fail", res, None, True)

    finally:
        if 'original_el_transcribe' in locals():
            resources.stt_client.primary.transcribe_audio = original_el_transcribe
        if 'original_sv_transcribe' in locals():
            resources.stt_client.fallback.transcribe_audio = original_sv_transcribe

if __name__ == "__main__":
    run_tests()
