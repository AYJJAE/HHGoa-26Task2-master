import sys
import os
import requests
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_fastapi_endpoints():
    url = "http://localhost:8000/api/voice_ask"
    
    # 1. Empty file
    try:
        print("Test 1: Empty file")
        res = requests.post(url, files={"audio": ("test.webm", b"", "audio/webm")}, data={"language": "en"})
        print(res.status_code, res.json())
    except Exception as e:
        print(e)
        
    # 2. Opus codec mime type
    try:
        print("\nTest 2: Opus codec")
        with open(os.path.join(os.path.dirname(os.path.dirname(__file__)), "test.wav"), "rb") as f:
            audio_bytes = f.read()
        res = requests.post(url, files={"audio": ("test.webm", audio_bytes, "audio/webm;codecs=opus")}, data={"language": "en"})
        print(res.status_code, res.json())
    except Exception as e:
        print(e)

if __name__ == "__main__":
    test_fastapi_endpoints()
