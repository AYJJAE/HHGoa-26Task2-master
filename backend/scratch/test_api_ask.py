import os
import sys
import json
from fastapi.testclient import TestClient

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from api.main import app

with TestClient(app) as client:
    # Test 1: text ask
    response = client.post("/api/ask", json={"query": "who is the prime minister of india"})
    print("Status Code:", response.status_code)
    data = response.json()
    print("Response JSON:")
    print(json.dumps(data, indent=2, ensure_ascii=False))

    assert data["status"] == "answered", f"Expected answered, got: {data}"
    assert data["refused"] is False
    assert data["grounded"] is True
    assert "Modi" in data["answer"] or "modi" in data["answer"].lower()
    print("\n>>> /api/ask SUCCESS! Grounded answer returned with HIGH confidence.")
