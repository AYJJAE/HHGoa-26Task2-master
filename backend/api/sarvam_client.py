import os
import requests
import tempfile
import base64
from typing import Optional, Tuple

class SarvamClient:
    def __init__(self):
        self.api_key = os.environ.get("SARVAM_API_KEY", "")
        self.base_url = "https://api.sarvam.ai"
        if not self.api_key:
            print("WARNING: SARVAM_API_KEY not set in environment. STT will return mock responses.")
            print("         Add SARVAM_API_KEY to your .env file for real speech-to-text.")

    def transcribe_audio(self, audio_bytes: bytes, filename: str = "audio.wav") -> Tuple[bool, str]:
        """
        Sends audio to Sarvam STT API and returns (success, transcribed_text).
        Falls back to a mock response only if no API key is set at all.
        """
        if not self.api_key:
            print("WARNING: No SARVAM_API_KEY found. Mocking STT response.")
            return True, "This is a mocked transcription because the Sarvam API key is missing. What is the capital of India?"
            
        # Write bytes to a temporary file since requests likes file objects for multipart
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name
            
        try:
            url = f"{self.base_url}/speech-to-text" # Using the base STT endpoint
            headers = {
                "api-subscription-key": self.api_key
            }
            
            # Using multipart/form-data
            with open(tmp_path, "rb") as f:
                files = {"file": (filename, f, "audio/wav")}
                # Depending on exact Sarvam API requirements, we might need model or language
                data = {"model": "saarika:v2.5"} 
                
                response = requests.post(url, headers=headers, files=files, data=data, timeout=10)
                
            if response.status_code == 200:
                result = response.json()
                return True, result.get("transcript", "")
            else:
                print(f"Sarvam STT Error: {response.status_code} - {response.text}")
                # Graceful degradation
                return False, "Could not transcribe audio due to API error."
                
        except Exception as e:
            print(f"Exception calling Sarvam STT: {e}")
            return False, "Could not transcribe audio due to internal error."
            
        finally:
            # Cleanup temp file
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
