import io
import sys
import os
import csv
import pytest
# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock, mock_open
from ASR_Module import asr_api


client = TestClient(asr_api.app)


@pytest.fixture(autouse=True)
def cleanup_files():
    """Clean up any test files before and after tests."""
    yield
    for folder in ["wav", "text"]:
        if os.path.exists(folder):
            for f in os.listdir(folder):
                if f.startswith("testsession"):
                    os.remove(os.path.join(folder, f))
    if os.path.exists("metadata.csv"):
        os.remove("metadata.csv")


def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_get_next_index_creates_new_metadata(tmp_path):
    # create a fake metadata.csv
    metadata_path = tmp_path / "metadata.csv"
    with open(metadata_path, "w", encoding="utf-8") as f:
        f.write("testsession|testsession_1.wav|hello world\n")
        f.write("testsession|testsession_3.wav|hi again\n")

    os.chdir(tmp_path)  # force cwd so asr_api finds this metadata
    idx = asr_api.get_next_index("testsession")
    assert idx == 4

@patch("ASR_Module.asr_api.open", new_callable=mock_open)   # use mock_open, not MagicMock
@patch("ASR_Module.asr_api.sf.read")
@patch("ASR_Module.asr_api.AudioSegment")
@patch("ASR_Module.asr_api.pipe")
def test_transcribe_endpoint(mock_pipe, mock_audio_segment, mock_sf_read, mock_file_open, tmp_path):
    # Mock pipe result
    mock_pipe.return_value = {"text": "hello test"}

    # Mock AudioSegment
    mock_audio = MagicMock()
    mock_audio.export = MagicMock()
    mock_audio.set_channels.return_value = mock_audio
    mock_audio.set_frame_rate.return_value = mock_audio
    mock_audio.set_sample_width.return_value = mock_audio
    mock_audio_segment.from_file.return_value = mock_audio

    # Mock sf.read to avoid real decoding
    mock_sf_read.return_value = (MagicMock(), 16000)

    # Dummy WebM data
    fake_audio = io.BytesIO(b"fake webm data")
    files = {"file": ("test.webm", fake_audio, "audio/webm")}
    data = {"session_id": "testsession"}

    response = client.post("/transcribe", files=files, data=data)

    assert response.status_code == 200
    assert response.json()["transcription"] == "hello test"

    # Assert that a text file would have been opened for writing
    mock_file_open.assert_any_call("text\\testsession_1.txt", "w", encoding="utf-8")