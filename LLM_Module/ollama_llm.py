from typing import TypedDict, Optional
from langchain_ollama import OllamaLLM
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langgraph.graph import StateGraph
from langgraph.checkpoint.memory import MemorySaver
# cloud api
from langchain_google_genai import ChatGoogleGenerativeAI

import unicodedata
import os
from dotenv import load_dotenv
import re
import datetime
import time
import random
import string
from station import FareSystem, find_station, plan_route, lines_data, rules_facilities_kb

# ==== INIT & GLOBALS ====
kiosk_station = "Kelana Jaya"
lang_instruction = "en" #en (English) or ms (Malay)
fare_system = FareSystem("Fare.csv", from_station=kiosk_station)
load_dotenv()

#llm = OllamaLLM(model="llama3:8b")
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",  # or gemini-1.5-flash for faster inference
    temperature=0,
    max_output_tokens=512,
    convert_system_message_to_human=True,
    google_api_key = os.getenv("GOOGLE_API_KEY")
)
memory = ConversationBufferMemory(input_key="input", memory_key="history")

SESSION_LOCKS: dict = {}

CONFIRM_TOKENS_EN = {"yes", "yep", "ok", "okay", "confirm", "buy", "sure"}
CANCEL_TOKENS_EN  = {"no", "nope", "not yet", "later", "cancel", "stop", "back"}

CONFIRM_TOKENS_MS = {"ya", "yao", "yeah", "yes", "sah", "betul", "ok", "okey", "oke", "confirm", "boleh", "setuju"}
CANCEL_TOKENS_MS  = {"tidak", "tak", "batal", "jangan", "nanti", "tidaklah", "stop", "balik"}

all_stations = [station.lower() for line, stations in lines_data.items() for station in stations]
# Define stopwords
STOPWORDS_MS = {
    "saya", "aku", "kami", "anda", "dia",
    "nak", "mahu", "ingin", "pergi", "ke", "dari", "balik",
    "turun", "naik", "dengan",
    "tiket", "harga", "tambang", "stesen", "lrt", "mrt", "monorel", "kereta", "api"
}

STOPWORDS_EN = {
    "i", "me", "we", "you", "they",
    "want", "would", "like", "need",
    "to", "go", "get", "take", "ride", "travel",
    "from", "to", "at", "with",
    "ticket", "fare", "station", "lrt", "mrt", "monorail", "train"
}

# ==== HELPERS ====

# json response builder for api
def build_json_response(
    text: str,
    query_type: str,
    session_id: str,
    ticket_details: dict = None,
    route_details: dict = None
):
    return {
        "text": text,
        "ticket_details": ticket_details or {
            "session_id": "",
            "ticket_id": "",
            "from_station": "",
            "to_station": "",
            "fare": "",
            "interchange": "",
            "datetime": ""
        },
        "route_details": route_details or {
            "station_lines": [],
            "interchanges": []
        },
        "query_type": query_type
    }

def normalize_user_input(text: str) -> str:
    """
    Normalize incoming user input (typed or transcribed).
    - Lowercases everything
    - Removes hidden unicode artifacts
    - Strips punctuation (except alphanumeric & spaces)
    - Collapses multiple spaces
    """
    if not text:
        return ""

    # Unicode normalization (removes hidden accents, weird tokens)
    text = unicodedata.normalize("NFKC", text)

    # Lowercase
    text = text.lower()

    # Remove non-alphanumeric except spaces
    text = re.sub(r"[^a-z0-9\s]", " ", text)

    # Collapse multiple spaces
    text = re.sub(r"\s+", " ", text).strip()

    return text

# process the user input for better extraction station matching
def preprocess_for_station_matching(text: str, lang: str) -> str:
    if not text:
        return ""

    text = text.lower()
    tokens = text.split()

    # Only remove stopwords if at least 2 tokens remain after removal
    stopwords = STOPWORDS_MS if lang == "ms" else STOPWORDS_EN
    filtered_tokens = [t for t in tokens if t not in stopwords]
    if len(filtered_tokens) >= 2:
        tokens = filtered_tokens

    # Collapse whitespace
    text = " ".join(tokens).strip()

    # Tokenize and filter to station-like candidates (full word match only)
    station_names_lower = [s.lower() for s in all_stations]
    candidates = [token for token in tokens if token in station_names_lower]

    # If we found candidates, keep only them
    if candidates:
        return " ".join(candidates)

    # Otherwise, fallback to cleaned text
    return text

# generate a random ticket ID
def generate_ticket_id(length=6):
    return ''.join(random.choices(string.ascii_uppercase + string.digits, k=length))

# return a per-session lock dict; create if missing
def get_session_lock(session_id: str):
    if session_id not in SESSION_LOCKS:
        SESSION_LOCKS[session_id] = {"locked": False, "station": None, "fare": None, "interchange": None, "time": None, "station_lines": [], "ordered_inters": []}
    return SESSION_LOCKS[session_id]

# language-safe confirmation / cancellation checks
def is_confirmation_text(text: str, lang: str = "en") -> bool:
    tokens = re.findall(r"\w+", (text or "").lower())
    tset = set(tokens)
    if lang == "ms":
        return bool(tset & CONFIRM_TOKENS_MS)
    return bool(tset & CONFIRM_TOKENS_EN)

def is_cancellation_text(text: str, lang: str = "en") -> bool:
    tokens = re.findall(r"\w+", (text or "").lower())
    tset = set(tokens)
    if lang == "ms":
        return bool(tset & CANCEL_TOKENS_MS)
    return bool(tset & CANCEL_TOKENS_EN)

CANCEL_MESSAGES = {
    "en": "Okay, please let me know if you need anything else.",
    "ms": "Baik, sila beritahu saya jika anda memerlukan apa-apa lagi."
}

ticket_prompt = PromptTemplate(
    input_variables=[
        "history", "input", "kiosk_station", "fare_info",
        "destination", "interchange_note", "time",
        "ticket_status", "language", "input_contains_exact_station"
    ],
    template="""
You are RapidKL Transit AI assistant.

Kiosk Station: {kiosk_station}
Conversation so far: {history}

Fare Information: {fare_info}
Destination Candidate: {destination}
Interchange Information: {interchange_note}
Estimated Travel Time: {time}
Ticket Status: {ticket_status}
Input Contains Exact Station: {input_contains_exact_station}

Guidelines:
1. Always use the **Fare Information**, **Destination Station**, and **Interchange Information** exactly as provided. Do NOT invent, modify, or guess new stations, fares, or times.
2. If ticket_status is "candidate" AND the station is only a candidate (not yet locked):
    - If Input Contains Exact Station is "False": Say: "The station is not recognized in the system. Do you mean: {destination}?"
    - Say: "I found a possible match: {destination}. It costs RM{fare_info}. Do you mean this station? Yes or No?"  
    - If user says "yes" → confirm and lock the destination, then move to Purchase Confirmation Phase.  
    - If user says "no" → politely cancel the request.  
3. If ticket_status is "yes":  
    - Assume the ticket purchase is confirmed.  
    - Only say: "Okay, generating your ticket now..."  
    - Do NOT issue or create ticket IDs yourself (the backend will handle it).
4. Keep responses short, polite, and always in the user’s language.  
5. IMPORTANT: Always respond in **{language}** only and do NOT switch to other languages.

Question: {input}
"""
)


# qna agent prompt
qna_prompt = PromptTemplate(
    input_variables=["history", "input", "kiosk_station", "fare_info", "destination", "interchange_note", "language", "time", "line", "stops", "available_stations"],
    template="""
You are RapidKL Transit AI assistant.
Answer user's transport-related questions briefly and politely in **{language}**. Do not translate to other languages.
Use the Fare Info and Referred Station below exactly when referencing fares. Just say "don't know" when no information is available.
If the question is related to KLIA Ekspres/Transit, mention that the service departs from KL Sentral or Putrajaya Sentral respectively.

Current kiosk station: {kiosk_station}
Conversation so far: {history}

Fare Info: {fare_info}
Referred Station: {destination}
Referred Station Line: {line}
Interchange Info: {interchange_note}
Estimated Travel Time to Referred Station: {time} minutes
Number of stops to Referred Station: {stops}
Available Stations: {available_stations}
Question: {input}
"""
)

# ==== STATE TYPE (add session_id) ====
class KioskState(TypedDict):
    input: str
    history: str
    output: str
    route: str
    session_id: str
    json: dict
    
# ==== AGENT NODES ====
def router_node(state: KioskState) -> KioskState:
    text = state["input"].lower()
    session_id = state.get("session_id", "default")
    lock = get_session_lock(session_id)

    user_lang = lang_instruction

    # If ticket flow already started (station locked) → confirmation/cancellation handled by ticket_agent
    if lock["station"] or lock.get("candidate_station"):
        if is_confirmation_text(text, user_lang) or is_cancellation_text(text, user_lang):
            state["route"] = "ticket_agent"
            print(f"[ROUTER] Routing to: {state['route']} for input: {state['input']} (confirmation/cancellation) session={session_id} lock={lock}")
            return state

    # === Intent keywords (regex-based) ===
    ROUTE_KEYWORDS = [
        r"\bpath\b", r"\broute\b", r"how to get", r"how do i go", r"how do i reach",
        r"how can i go", r"best route", r"fastest route", r"directions?", r"how do i get to",
        r"\bjalan\b", r"cara pergi", r"macam mana nak ke", r"\bmap\b",
        r"bagaimana untuk ke", r"arah?", r"\blaluan\b", r"\bke mana\b"
    ]
    TICKET_KEYWORDS = [
        r"\bbuy\b", r"\bticket\b", r"\btiket\b", r"\bfare\b",
        r"\bprice\b", r"how much", r"\bi want to go\b",
        r"\bke\b", r"\bpergi\b", r"berapa tambang", r"harga tiket",
        r"nak beli", r"saya nak beli tiket"
    ]

    if any(re.search(pattern, text) for pattern in ROUTE_KEYWORDS):
        state["route"] = "route_planning"
    elif any(re.search(pattern, text) for pattern in TICKET_KEYWORDS):
        state["route"] = "ticket_agent"
    else:
        state["route"] = "qna_agent"

    print(f"[ROUTER] Routing to: {state['route']} for input: {state['input']} session={session_id}")
    return state

def ticket_agent_node(state: KioskState) -> KioskState:
    print(f"[AGENT] Running ticket_agent_node for: {state['input']}")
    session_id = state.get("session_id", "default")
    lock = get_session_lock(session_id)

    user_input = normalize_user_input(state["input"])
    user_lang = lang_instruction
    language = "Malay" if user_lang == "ms" else "English"
    history = state.get("history", "")
    available_stations = all_stations
    input_contains_exact_station = any(station in normalize_user_input(user_input) for station in available_stations)

    fare_info = "None"
    destination = "None"
    interchange_note = "None"
    time = ""
    ticket_status = "no"

    # -------------------------------
    # Case 1: Handle confirmation / cancellation for candidate station
    # -------------------------------
    if lock.get("candidate_station") and not lock.get("locked", False):
        if is_confirmation_text(user_input, user_lang):
            # Promote candidate -> confirmed station
            lock["station"] = lock["candidate_station"]
            lock["locked"] = True
            lock.pop("candidate_station", None)
            print(f"[AGENT] session={session_id} confirmation detected -> locking station={lock['station']}")
        elif is_cancellation_text(user_input, user_lang):
            # Reset everything
            SESSION_LOCKS[session_id] = {
                "locked": False, "station": None, "candidate_station": None,
                "fare": None, "interchange": None, "time": None,
                "station_lines": [], "ordered_inters": []
            }
            reply_text = CANCEL_MESSAGES.get(user_lang, CANCEL_MESSAGES["en"])
            return {
                "input": user_input,
                "history": history,
                "output": reply_text,
                "route": "",
                "json": build_json_response(
                    text=reply_text,
                    query_type="ticket_agent",
                    session_id=session_id
                )
            }
        else:
            clean_input = preprocess_for_station_matching(user_input, user_lang)
            station, _, _ = find_station(clean_input)

            if station and station != lock.get("candidate_station"):
                # Replace candidate with new station
                fare, _, _ = fare_system.get_fare(station)
                dt = datetime.datetime.now()
                est_time, _, = fare_system.estimate_travel_time(station, dt)
                _, _, station_lines, ordered_inters = plan_route(kiosk_station, station, fare_system)

                interchange_note = f"Includes interchanges at {', '.join(ordered_inters)}." if ordered_inters else ""

                lock.update({
                    "candidate_station": station,
                    "fare": fare,
                    "interchange": interchange_note,
                    "time": est_time,
                    "station_lines": station_lines,
                    "ordered_inters": ordered_inters
                })
                print(f"[AGENT] session={session_id} switched candidate_station -> {station}")
            
            ticket_status = "candidate"

    # -------------------------------
    # Case 2: Detect new candidate if nothing in lock
    # -------------------------------
    if lock.get("station") is None and lock.get("candidate_station") is None:
        clean_input = preprocess_for_station_matching(user_input, user_lang)
        print(f"[AGENT] Cleaned input for station matching: {repr(clean_input)}")
        station, _, _ = find_station(clean_input)
        print(f"[AGENT] Extracted station: {station}")

        if station:
            fare, _, _ = fare_system.get_fare(station)
            dt = datetime.datetime.now()
            est_time, _, = fare_system.estimate_travel_time(station, dt)
            _, _, station_lines, ordered_inters = plan_route(kiosk_station, station, fare_system)

            interchange_note = f"Includes interchanges at {', '.join(ordered_inters)}." if ordered_inters else ""

            lock.update({
                "candidate_station": station,
                "fare": fare,
                "interchange": interchange_note,
                "time": est_time,
                "station_lines": station_lines,
                "ordered_inters": ordered_inters
            })
            print(f"[AGENT] session={session_id} set candidate_station -> {lock}")
        else:
            fare_info = "None"

    # -------------------------------
    # Case 3: If confirmed purchase (locked + user said yes)
    # -------------------------------
    if lock.get("locked", False) and lock.get("station") and is_confirmation_text(user_input, user_lang):
        ticket_id = generate_ticket_id()
        current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        ticket_text_en = (
            f"Your ticket is issued! From {kiosk_station} to {lock['station']} "
            f"with a fare of RM{lock['fare']:.2f}. {lock['interchange']} "
            f"The estimated travel time is around {lock['time']} minutes. "
            f"Please download this PDF ticket, enjoy your journey!"
        )
        ticket_text_ms = (
            f"Tiket anda sudah dikeluarkan. Dari {kiosk_station} ke {lock['station']} "
            f"dengan tambang RM{lock['fare']:.2f}. {lock['interchange']} "
            f"Anggaran masa perjalanan adalah sekitar {lock['time']} minit. "
            f"Sila muat turun tiket PDF ini, selamat menikmati perjalanan anda!"
        )
        ticket_text = ticket_text_en if user_lang != "ms" else ticket_text_ms

        ticket_details = {
            "session_id": session_id,
            "ticket_id": ticket_id,
            "from_station": kiosk_station,
            "to_station": lock["station"],
            "fare": f"RM{lock['fare']:.2f}",
            "interchange": lock["interchange"],
            "datetime": current_time
        }

        route_details = {
            "station_lines": lock["station_lines"],
            "interchanges": lock["ordered_inters"]
        }

        memory.save_context({"input": "[SYSTEM] Transaction Completed"},
                            {"output": f"--- Ticket transaction completed: {ticket_id} ---"})
        memory.clear()

        SESSION_LOCKS[session_id] = {
            "locked": False, "station": None, "candidate_station": None,
            "fare": None, "interchange": None, "time": None,
            "station_lines": [], "ordered_inters": []
        }
        print(f"[AGENT] session={session_id} ticket issued -> {ticket_id}")

        return {
            "input": user_input,
            "history": "",
            "output": ticket_text,
            "route": "",
            "json": build_json_response(
                text=ticket_text,
                query_type="ticket_agent",
                session_id=session_id,
                ticket_details=ticket_details,
                route_details=route_details
            )
        }

    # -------------------------------
    # Case 4: Pass state to LLM for clarification / confirmation
    # -------------------------------
    if lock.get("candidate_station"):
        ticket_status = "candidate"
        fare_info = f"RM{lock['fare']:.2f}" if lock.get("fare") else ""
        destination = lock["candidate_station"]
        time = f"{lock['time']}" if lock.get("time") else ""
        interchange_note = lock["interchange"] or ""
    elif lock.get("locked", False) and lock.get("station"):
        ticket_status = "locked"
        fare_info = f"RM{lock['fare']:.2f}"
        destination = lock["station"]
        time = f"{lock['time']}"
        interchange_note = lock["interchange"] or ""

    reply = llm.invoke(ticket_prompt.format(
        history=history,
        input=user_input,
        kiosk_station=kiosk_station,
        fare_info=fare_info,
        destination=destination,
        interchange_note=interchange_note,
        time=time,
        ticket_status=ticket_status,
        language=language,
        input_contains_exact_station=input_contains_exact_station
    ))
    print(f"[AGENT] session={session_id} sent prompt (status={ticket_status}, language={language}) lock={lock}")

    reply_text = reply.content if hasattr(reply, "content") else str(reply)
    
    return {
        "input": user_input,
        "history": history,
        "output": reply_text,
        "route": "ticket_agent",
        "json": build_json_response(
            text=reply_text,
            query_type="ticket_agent",
            session_id=session_id
        )
    }



def qna_agent_node(state: KioskState) -> KioskState:
    print(f"[AGENT] Running qna_agent_node for: {state['input']}")
    user_input = normalize_user_input(state["input"])
    user_lang = lang_instruction
    language = "Malay" if user_lang == "ms" else "English"
    history = state.get("history", "")

    # === STEP 1: Try station/fare-based queries first ===
    fare_info = "None"
    interchange_note = "None"
    destination = "None"
    available_stations = all_stations
    clean_input = preprocess_for_station_matching(user_input, user_lang)
    station, line, _ = find_station(clean_input)
    
    if station:
        fare, _, _ = fare_system.get_fare(station)
        dt = datetime.datetime.now()
        time, breakdown, = fare_system.estimate_travel_time(station, dt)
        stops = breakdown.get("stops", 0)
        _, _, _,ordered_inters = plan_route(kiosk_station, station, fare_system)
        # Build note for LLM
        if ordered_inters:
            interchange_note = f"Includes interchanges at {', '.join(ordered_inters)}."
        else:
            interchange_note = ""
        fare_info = f"RM{fare:.2f}" if fare is not None else ""
        destination = f"{station}"
        station_line = f"{line}"
        time = f"{time}"
        
        # LLM handle queries
        reply = llm.invoke(qna_prompt.format(
            history=history,
            input=user_input,
            kiosk_station=kiosk_station,
            fare_info=fare_info,
            destination=destination,
            interchange_note=interchange_note,
            time=time,
            stops=stops,
            line=station_line,
            language=language,
            available_stations=", ".join(available_stations)
        ))
    
    else:
        # === STEP 2: Fall back to knowledge base ===
        kb_results = rules_facilities_kb.query(user_input, k=2)
        if kb_results:
            # Concatenate top-k answers for context
            kb_context = "\n".join([f"- {item['question']}: {item['answer']}" for item in kb_results])
            # Pass context into the LLM prompt
            reply = llm.invoke(
                f"{user_input}\n\nRelevant info:\n{kb_context}\n\nPlease give short and simple answers using the above context."
            )
        else:
            reply = "Sorry, I couldn't find relevant info."

    memory.clear()
    
    reply_text = reply.content if hasattr(reply, "content") else str(reply)
    
    return_data = {
        "input": user_input,
        "history": history,
        "output": reply_text,
        "route": "",
        "json": build_json_response(
            text=reply_text,
            query_type="qna_agent",
            session_id=state["session_id"],
            ticket_details=None,
            route_details=None
        )
    }
    
    print(f"### DEBUG [qna_agent_node] Returning={return_data}")

    return return_data

def route_planning_node(state: KioskState) -> KioskState:
    print(f"[AGENT] Running route_planning_node for: {state['input']}")
    user_input = normalize_user_input(state["input"])
    user_lang = lang_instruction
    history = state.get("history", "")
    
    clean_input = preprocess_for_station_matching(user_input, user_lang)
    dest_station, _, _ = find_station(clean_input)
    
    if not dest_station:
        reply_en = "I couldn’t find the destination station. Please try again."
        reply_ms = "Saya tidak dapat mencari stesen destinasi. Sila cuba lagi."
        reply_text = reply_en if user_lang != "ms" else reply_ms
        station_lines, interchanges = [], []
    else:
        route_text_en, route_text_ms, station_lines, interchanges = plan_route(
            kiosk_station, dest_station, fare_system
        )
        reply_text = route_text_en if user_lang != "ms" else route_text_ms

    memory.clear()
    
    return_data = {
        "input": user_input,
        "history": history,
        "output": reply_text,
        "route": "",
        "json": build_json_response(
            text=reply_text,
            query_type="route_planning",
            session_id=state["session_id"],
            ticket_details=None,
            route_details={
                "station_lines": station_lines,
                "interchanges": interchanges
            }
        )
    }
    print(f"### DEBUG [route_planning_node] Returning={return_data}")
    return return_data


# ==== BUILD STATEGRAPH ====
workflow = StateGraph(KioskState)

workflow.add_node("router", router_node)
workflow.add_node("ticket_agent", ticket_agent_node)
workflow.add_node("qna_agent", qna_agent_node)
workflow.add_node("route_planning", route_planning_node)

workflow.set_entry_point("router")
workflow.add_conditional_edges(
    "router",
    lambda state: state["route"],
    {
        "ticket_agent": "ticket_agent",
        "qna_agent": "qna_agent",
        "route_planning": "route_planning"
    }
)

checkpointer = MemorySaver()
app = workflow.compile(checkpointer=checkpointer)

# ==== RUN FUNCTION ====
def run_llm(user_message: str, session_id: str):
    # sanitize input
    clean_message = normalize_user_input(user_message)
    state = {
        "input": clean_message,
        "history": memory.load_memory_variables({}).get("history", ""),
        "output": "",
        "route": "",
        "session_id": session_id
    }
    print(f"\n### DEBUG: Incoming state={state}")
    start_time = time.time()
    result = app.invoke(
        state,
        config={"configurable": {"thread_id": session_id}}
    )
    end_time = time.time()
    print(f"### DEBUG: Raw result from workflow={result}")
    print(f"### DEBUG: LLM response time: {end_time - start_time:.2f} seconds")

    # 🔧 Normalize output so memory always gets plain string
    raw_output = result.get("output", "")
    if hasattr(raw_output, "content"):   # AIMessage or similar
        output_text = raw_output.content
    else:
        output_text = str(raw_output)

    # Only persist if it's ticket agent
    if result["route"] == "ticket_agent":
        memory.save_context({"input": user_message}, {"output": output_text})

    # Fallback JSON
    json_data = result.get("json")
    if not json_data:
        json_data = build_json_response(
            text=output_text,
            query_type=result.get("route", "unknown"),
            session_id=session_id,
            ticket_details=None,
            route_details=None
        )

    # Always ensure keys exist
    json_data.setdefault("ticket_details", {
        "session_id": "",
        "ticket_id": "",
        "from_station": "",
        "to_station": "",
        "fare": "",
        "interchange": "",
        "datetime": ""
    })
    json_data.setdefault("route_details", {
        "station_lines": [],
        "interchanges": []
    })

    print(f"### DEBUG: Final JSON response={json_data}\n")
    return json_data
