import sys
import os
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Patch the station module before importing ollama_llm
mock_station = MagicMock()
mock_station.KnowledgeBase = MagicMock()
mock_station.FareSystem = MagicMock()
mock_station.find_station = MagicMock()
mock_station.plan_route = MagicMock()
mock_station.lines_data = {}
mock_station.rules_facilities_kb = MagicMock()
sys.modules["station"] = mock_station

import pytest
import ollama_llm

# ==== Helper Function Tests ====

def test_build_json_response_defaults():
    resp = ollama_llm.build_json_response("Hello", "qna_agent", "sess1")
    assert resp["text"] == "Hello"
    assert resp["query_type"] == "qna_agent"
    assert "ticket_details" in resp
    assert "route_details" in resp

def test_preprocess_for_station_matching_en_removes_stopwords():
    text = "I want to go to KL Sentral"
    result = ollama_llm.preprocess_for_station_matching(text, "en")
    assert "kl sentral" in result

def test_preprocess_for_station_matching_ms_removes_stopwords():
    text = "Saya ingin pergi ke KL Sentral"
    result = ollama_llm.preprocess_for_station_matching(text, "ms")
    assert "kl sentral" in result

def test_generate_ticket_id_length():
    ticket_id = ollama_llm.generate_ticket_id()
    assert len(ticket_id) == 6

def test_get_session_lock_creates_and_returns():
    session_id = "testsession"
    lock = ollama_llm.get_session_lock(session_id)
    assert isinstance(lock, dict)
    assert "locked" in lock

def test_is_confirmation_text_en():
    assert ollama_llm.is_confirmation_text("Yes, please", "en")
    assert not ollama_llm.is_confirmation_text("No, thanks", "en")

def test_is_confirmation_text_ms():
    assert ollama_llm.is_confirmation_text("Ya, boleh", "ms")
    assert not ollama_llm.is_confirmation_text("Tidak", "ms")

def test_is_cancellation_text_en():
    assert ollama_llm.is_cancellation_text("No, thanks", "en")
    assert not ollama_llm.is_cancellation_text("Yes", "en")

def test_is_cancellation_text_ms():
    assert ollama_llm.is_cancellation_text("Batal", "ms")
    assert not ollama_llm.is_cancellation_text("Ya", "ms")

# ==== Agent Node Tests ====

def test_router_node_ticket():
    state = {
        "input": "I want to buy a ticket to KL Sentral",
        "history": "",
        "output": "",
        "route": "",
        "session_id": "sess6",
        "json": {}
    }
    result = ollama_llm.router_node(state)
    assert result["route"] == "ticket_agent"

def test_router_node_route():
    state = {
        "input": "How do I go to KL Sentral?",
        "history": "",
        "output": "",
        "route": "",
        "session_id": "sess7",
        "json": {}
    }
    result = ollama_llm.router_node(state)
    assert result["route"] == "route_planning"

def test_router_node_qna():
    state = {
        "input": "What are the facilities at KL Sentral?",
        "history": "",
        "output": "",
        "route": "",
        "session_id": "sess8",
        "json": {}
    }
    result = ollama_llm.router_node(state)
    assert result["route"] == "qna_agent"

def test_ticket_agent_node_candidate_confirmation():
    session_id = "sess1"
    lock = ollama_llm.get_session_lock(session_id)
    lock["candidate_station"] = "KL Sentral"
    lock["locked"] = False
    # Pre-populate fare info so formatting works
    lock["fare"] = 2.50
    lock["time"] = 15
    lock["interchange"] = ""

    with patch.object(ollama_llm, "find_station") as mock_find_station, \
         patch.object(ollama_llm, "llm") as mock_llm:
        mock_find_station.return_value = ("KL Sentral", None, None)
        mock_llm.invoke.return_value = "I found a possible match: KL Sentral. It costs RM2.50. Do you mean this station?"

        fare_system_mock = MagicMock()
        fare_system_mock.get_fare.return_value = (2.50, None, None)
        fare_system_mock.estimate_travel_time.return_value = (15, None)
        ollama_llm.fare_system = fare_system_mock
        ollama_llm.plan_route = MagicMock(return_value=("route_en", "route_ms", ["Kelana Jaya", "KL Sentral"], []))

        state = {
            "input": "Yes",
            "history": "",
            "output": "",
            "route": "",
            "session_id": session_id,
            "json": {}
        }
        result = ollama_llm.ticket_agent_node(state)
        assert "KL Sentral" in result["json"]["text"]
        assert "RM2.50" in result["json"]["text"]


def test_ticket_agent_node_new_candidate():
    session_id = "sess2"
    lock = ollama_llm.get_session_lock(session_id)
    lock["station"] = None
    lock["candidate_station"] = None

    with patch.object(ollama_llm, "find_station") as mock_find_station, \
         patch.object(ollama_llm, "llm") as mock_llm:
        mock_find_station.return_value = ("KL Sentral", None, None)
        mock_llm.invoke.return_value = "I found a possible match: KL Sentral. It costs RM2.50. Do you mean this station?"

        fare_system_mock = MagicMock()
        fare_system_mock.get_fare.return_value = (2.50, None, None)
        fare_system_mock.estimate_travel_time.return_value = (15, None)
        ollama_llm.fare_system = fare_system_mock
        ollama_llm.plan_route = MagicMock(return_value=("route_en", "route_ms", ["Kelana Jaya", "KL Sentral"], []))

        state = {
            "input": "I want to go to KL Sentral",
            "history": "",
            "output": "",
            "route": "",
            "session_id": session_id,
            "json": {}
        }
        result = ollama_llm.ticket_agent_node(state)
        assert "KL Sentral" in result["json"]["text"]
        assert "RM2.50" in result["json"]["text"]


def test_qna_agent_node_station_found():
    with patch.object(ollama_llm, "find_station") as mock_find_station, \
         patch.object(ollama_llm, "llm") as mock_llm:
        mock_find_station.return_value = ("KL Sentral", "Kelana Jaya", None)
        mock_llm.invoke.return_value = "KL Sentral is on the Kelana Jaya line."

        fare_system_mock = MagicMock()
        fare_system_mock.get_fare.return_value = (2.50, None, None)
        fare_system_mock.estimate_travel_time.return_value = (15, {"stops": 5})
        ollama_llm.fare_system = fare_system_mock

        state = {
            "input": "Tell me about KL Sentral",
            "history": "",
            "output": "",
            "route": "",
            "session_id": "sess3",
            "json": {}
        }
        result = ollama_llm.qna_agent_node(state)
        text = result["json"]["text"]
        assert "KL Sentral" in text
        assert ("RM2.50" in text) or ("Kelana Jaya" in text)  # Fare may or may not be mentioned depending on LLM response

def test_route_planning_node_valid_station():
    with patch.object(ollama_llm, "find_station") as mock_find_station:
        mock_find_station.return_value = ("KL Sentral", None, None)
        ollama_llm.plan_route = MagicMock(return_value=(
            "Take the Kelana Jaya line to KL Sentral.",
            "Naik laluan Kelana Jaya ke KL Sentral.",
            ["Kelana Jaya", "KL Sentral"],
            ["Masjid Jamek"]
        ))
        fare_system_mock = MagicMock()
        ollama_llm.fare_system = fare_system_mock

        state = {
            "input": "How do I get to KL Sentral?",
            "history": "",
            "output": "",
            "route": "",
            "session_id": "sess4",
            "json": {}
        }
        result = ollama_llm.route_planning_node(state)
        text = result["json"]["text"]
        assert "KL Sentral" in text
        assert "Kelana Jaya" in result["json"]["route_details"]["station_lines"]
        assert "Masjid Jamek" in result["json"]["route_details"]["interchanges"]
        
def test_route_planning_node_invalid_station():
    with patch.object(ollama_llm, "find_station") as mock_find_station:
        mock_find_station.return_value = (None, None, None)
        state = {
            "input": "How do I get to Atlantis?",
            "history": "",
            "output": "",
            "route": "",
            "session_id": "sess5",
            "json": {}
        }
        result = ollama_llm.route_planning_node(state)
        text = result["json"]["text"]
        assert "couldn’t find" in text or "tidak dapat" in text
        assert result["json"]["route_details"]["station_lines"] == []
        assert result["json"]["route_details"]["interchanges"] == []

# ==== Additional Crucial Tests ====

def test_get_session_lock_new_id():
    session_id = "newsession"
    lock = ollama_llm.get_session_lock(session_id)
    assert isinstance(lock, dict)
    assert lock["station"] is None
    assert lock["locked"] is False

def test_ticket_agent_node_cancellation_flow():
    session_id = "sess_cancel"
    lock = ollama_llm.get_session_lock(session_id)
    lock["candidate_station"] = "KL Sentral"
    lock["locked"] = False
    with patch.object(ollama_llm, "find_station"), patch.object(ollama_llm, "llm"):
        state = {
            "input": "No, cancel",
            "history": "",
            "output": "",
            "route": "",
            "session_id": session_id,
            "json": {}
        }
        result = ollama_llm.ticket_agent_node(state)
        assert "Okay, please let me know" in result["json"]["text"]
        lock = ollama_llm.get_session_lock(session_id)
        assert lock["station"] is None
        assert lock["candidate_station"] is None
        assert lock["locked"] is False

def test_qna_agent_node_kb_fallback():
    with patch.object(ollama_llm, "find_station") as mock_find_station, \
            patch.object(ollama_llm, "llm") as mock_llm, \
            patch.object(ollama_llm, "rules_facilities_kb") as mock_kb:
        mock_find_station.return_value = (None, None, None)
        mock_kb.query.return_value = [
            {"question": "Can I bring a bicycle?", "answer": "Yes, on weekends."}
        ]
        mock_llm.invoke.return_value = "Yes, you can bring a bicycle on weekends."
        state = {
            "input": "Can I bring a bicycle?",
            "history": "",
            "output": "",
            "route": "",
            "session_id": "sess_kb",
            "json": {}
        }
        result = ollama_llm.qna_agent_node(state)
        assert "bicycle" in result["json"]["text"]

def test_qna_agent_node_kb_no_result():
    with patch.object(ollama_llm, "find_station") as mock_find_station, \
            patch.object(ollama_llm, "llm") as mock_llm, \
            patch.object(ollama_llm, "rules_facilities_kb") as mock_kb:
        mock_find_station.return_value = (None, None, None)
        mock_kb.query.return_value = []
        state = {
            "input": "Unknown question?",
            "history": "",
            "output": "",
            "route": "",
            "session_id": "sess_kb2",
            "json": {}
        }
        result = ollama_llm.qna_agent_node(state)
        assert "couldn't find" in result["json"]["text"].lower()

def test_ticket_agent_node_candidate_station_replacement():
    session_id = "sess_replace"
    lock = ollama_llm.get_session_lock(session_id)
    lock["candidate_station"] = "Old Station"
    lock["locked"] = False
    with patch.object(ollama_llm, "find_station") as mock_find_station, \
            patch.object(ollama_llm, "llm") as mock_llm:
        mock_find_station.return_value = ("New Station", None, None)
        fare_system_mock = MagicMock()
        fare_system_mock.get_fare.return_value = (3.00, None, None)
        fare_system_mock.estimate_travel_time.return_value = (20, None)
        ollama_llm.fare_system = fare_system_mock
        ollama_llm.plan_route = MagicMock(return_value=("route_en", "route_ms", ["Kelana Jaya", "New Station"], ["Masjid Jamek"]))
        state = {
            "input": "I want to go to New Station",
            "history": "",
            "output": "",
            "route": "",
            "session_id": session_id,
            "json": {}
        }
        result = ollama_llm.ticket_agent_node(state)
        lock = ollama_llm.get_session_lock(session_id)
        assert lock["candidate_station"] == "New Station"
        assert lock["fare"] == 3.00

def test_run_llm_ticket_agent():
    with patch.object(ollama_llm, "llm") as mock_llm, \
            patch.object(ollama_llm, "find_station") as mock_find_station:
        mock_llm.invoke.return_value = "Ticket issued."
        mock_find_station.return_value = ("KL Sentral", None, None)
        fare_system_mock = MagicMock()
        fare_system_mock.get_fare.return_value = (2.50, None, None)
        fare_system_mock.estimate_travel_time.return_value = (15, None)
        ollama_llm.fare_system = fare_system_mock
        ollama_llm.plan_route = MagicMock(return_value=("route_en", "route_ms", ["Kelana Jaya", "KL Sentral"], []))
        result = ollama_llm.run_llm("I want to buy a ticket to KL Sentral", "sess_runllm")
        assert result["query_type"] == "ticket_agent"
        assert "ticket_details" in result

def test_preprocess_for_station_matching_edge_cases():
    # Empty string
    assert ollama_llm.preprocess_for_station_matching("", "en") == ""
    # Non-existent station
    assert ollama_llm.preprocess_for_station_matching("Atlantis", "en") == "atlantis"

def test_is_confirmation_text_and_cancellation_text_edge_cases():
    assert ollama_llm.is_confirmation_text("YES!", "en")
    assert ollama_llm.is_confirmation_text("ok.", "en")
    assert ollama_llm.is_cancellation_text("No.", "en")
    assert ollama_llm.is_cancellation_text("BATAL!", "ms")

def test_router_node_confirmation_cancellation_routing():
    session_id = "sess_router"
    lock = ollama_llm.get_session_lock(session_id)
    lock["candidate_station"] = "KL Sentral"
    lock["locked"] = False
    state = {
        "input": "Yes",
        "history": "",
        "output": "",
        "route": "",
        "session_id": session_id,
        "json": {}
    }
    result = ollama_llm.router_node(state)
    assert result["route"] == "ticket_agent"

def test_route_planning_node_malay():
    ollama_llm.lang_instruction = "ms"
    with patch.object(ollama_llm, "find_station") as mock_find_station:
        mock_find_station.return_value = ("KL Sentral", None, None)
        ollama_llm.plan_route = MagicMock(return_value=(
            "Take the Kelana Jaya line to KL Sentral.",
            "Naik laluan Kelana Jaya ke KL Sentral.",
            ["Kelana Jaya", "KL Sentral"],
            ["Masjid Jamek"]
        ))
        fare_system_mock = MagicMock()
        ollama_llm.fare_system = fare_system_mock
        state = {
            "input": "Bagaimana untuk ke KL Sentral?",
            "history": "",
            "output": "",
            "route": "",
            "session_id": "sess_ms",
            "json": {}
        }
        result = ollama_llm.route_planning_node(state)
        text = result["json"]["text"]
        assert "Naik laluan Kelana Jaya ke KL Sentral." in text
        ollama_llm.lang_instruction = "en"  # Reset

def test_ticket_agent_node_purchase_confirmation():
    session_id = "sess_confirm"
    lock = ollama_llm.get_session_lock(session_id)
    lock["station"] = "KL Sentral"
    lock["locked"] = True
    lock["fare"] = 2.50
    lock["interchange"] = ""
    lock["time"] = 15
    lock["station_lines"] = ["Kelana Jaya", "KL Sentral"]
    lock["ordered_inters"] = []
    with patch.object(ollama_llm, "llm") as mock_llm:
        mock_llm.invoke.return_value = "Your ticket is issued!"
        state = {
            "input": "Yes",
            "history": "",
            "output": "",
            "route": "",
            "session_id": session_id,
            "json": {}
        }
        result = ollama_llm.ticket_agent_node(state)
        assert (
            "ticket issued" in result["json"]["text"].lower()
            or "your ticket is issued" in result["json"]["text"].lower()
            or "tiket anda sudah dikeluarkan" in result["json"]["text"].lower()
        )
