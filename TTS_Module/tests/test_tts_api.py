# tests/test_docker_api.py
import requests
import os
from pathlib import Path

BASE_URL = "http://localhost:8020"

def test_healthcheck():
    # FastAPI doesn't have a /health in your melottsms.py yet,
    # but you can add one for easier testing.
    response = requests.get(f"{BASE_URL}/docs")
    assert response.status_code == 200

def test_tts_infer():
    payload = {"text": "Selamat pagi", "speaker": "Husein"}
    response = requests.post(f"{BASE_URL}/infer", json=payload)

    assert response.status_code == 200
    assert response.headers["content-type"] == "audio/wav"

    # Save in a pytest-managed temp folder
    output_path = Path("TTS_Module/tests/test_output.wav")
    output_path.write_bytes(response.content)
    
    assert output_path.exists()
    assert output_path.stat().st_size > 100
