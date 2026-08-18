import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)

def test_voice_ask():
    audio_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "test.wav")
    with open(audio_path, "rb") as f:
        audio_bytes = f.read()
    
    response = client.post(
        "/api/voice_ask",
        files={"audio": ("test.wav", audio_bytes, "audio/wav")},
        data={"language": "en"}
    )
    print("STATUS:", response.status_code)
    print("RESPONSE:", response.json())

if __name__ == "__main__":
    test_voice_ask()
