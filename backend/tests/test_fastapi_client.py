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
        data={"language": "en"},
        headers={"Origin": "https://task2-horizonlabs.vercel.app"}
    )
    assert response.headers.get("access-control-allow-origin") == "https://task2-horizonlabs.vercel.app"
    print("STATUS:", response.status_code)
    print("RESPONSE:", response.json())

def test_cors_options_preflight():
    response = client.options(
        "/api/voice_ask",
        headers={
            "Origin": "https://task2-horizonlabs.vercel.app",
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "content-type"
        }
    )
    assert response.status_code == 200
    assert response.headers.get("access-control-allow-origin") == "https://task2-horizonlabs.vercel.app"
    assert "POST" in response.headers.get("access-control-allow-methods", "")

def test_cors_ask_post():
    response = client.post(
        "/api/ask",
        json={"query": "Hello"},
        headers={"Origin": "https://task2-horizonlabs.vercel.app"}
    )
    assert response.headers.get("access-control-allow-origin") == "https://task2-horizonlabs.vercel.app"

def test_cors_on_error():
    response = client.post(
        "/api/ask",
        json={"query": ""},
        headers={"Origin": "https://task2-horizonlabs.vercel.app"}
    )
    assert response.headers.get("access-control-allow-origin") == "https://task2-horizonlabs.vercel.app"

if __name__ == "__main__":
    test_cors_options_preflight()
    test_cors_ask_post()
    test_cors_on_error()
    test_voice_ask()
