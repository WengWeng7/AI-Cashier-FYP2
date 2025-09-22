# tests/test_llm_api.py
import sys
import os
from unittest.mock import MagicMock, patch
import pytest
from fastapi.testclient import TestClient

# -------------------------------------------------------------------
# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Patch external dependencies before importing llm_api
mock_station = MagicMock()
mock_station.KnowledgeBase = MagicMock()
mock_station.FareSystem = MagicMock()
mock_station.find_station = MagicMock()
mock_station.plan_route = MagicMock()
mock_station.lines_data = {}
mock_station.rules_facilities_kb = MagicMock()

mock_llm_ui_helpers = MagicMock()
mock_llm_ui_helpers.generate_ticket_pdf = MagicMock()

sys.modules["station"] = mock_station
sys.modules["llm_ui_helpers"] = mock_llm_ui_helpers

# -------------------------------------------------------------------
# Import the FastAPI app after mocks
from llm_api import app

client = TestClient(app)

# -------------------------------------------------------------------
# Tests
def test_root():
    r = client.get("/")
    assert r.status_code == 200
    assert "Welcome to AI Ticketing Kiosk API" in r.json()["message"]


@patch("llm_api.run_llm")
def test_chat_endpoint(mock_run_llm):
    mock_run_llm.return_value = {
        "text": "Hello passenger!",
        "ticket_details": {"from_station": "A", "to_station": "B", "fare": "2.50"},
        "route_details": {"station_lines": [["A", "B"]], "interchanges": []},
        "query_type": "route",
    }

    r = client.post("/chat", json={"user_message": "hi", "session_id": "abc123"})
    assert r.status_code == 200
    data = r.json()
    assert data["text"] == "Hello passenger!"
    assert data["ticket_details"]["from_station"] == "A"


def test_set_language_valid():
    r = client.post("/set_language", json={"language": "ms"})
    assert r.status_code == 200
    assert "Language updated" in r.json()["message"]


def test_set_language_invalid():
    r = client.post("/set_language", json={"language": "xx"})
    assert r.status_code == 200
    assert "error" in r.json()


def test_set_station():
    r = client.post("/set_station", json={"station_name": "KLCC"})
    assert r.status_code == 200
    assert "Kiosk station updated to KLCC" in r.json()["message"]


@patch("llm_api.generate_ticket_pdf")
def test_generate_ticket(mock_pdf):
    mock_pdf.return_value = b"%PDF-1.4 fake pdf bytes"
    ticket = {
        "session_id": "s1",
        "ticket_id": "t1",
        "from_station": "A",
        "to_station": "B",
        "fare": "2.50",
        "interchange": "",
        "datetime": "2025-09-15T12:00",
    }

    r = client.post("/generate_ticket", json=ticket)
    assert r.status_code == 200
    assert r.headers["content-type"] == "application/pdf"
    assert r.content.startswith(b"%PDF-1.4")


def test_route_map_with_coords():
    payload = {
        "station_lines": [["StationA", "StationB"]],
        "interchanges": [],
    }
    r = client.post("/route_map", json=payload)
    assert r.status_code == 200
    data = r.json()
    # Either valid route or error response
    assert "stations" in data or "error" in data
