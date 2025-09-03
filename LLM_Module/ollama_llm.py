from typing import TypedDict, Optional
from langchain_ollama import OllamaLLM
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langgraph.graph import StateGraph
from langgraph.checkpoint.memory import MemorySaver

import re
import datetime
import random
import string
from langdetect import detect
from station import FareSystem, find_station, plan_route

# ==== INIT & GLOBALS ====
kiosk_station = "Kelana Jaya"
fare_system = FareSystem("Fare.csv", from_station=kiosk_station)

llm = OllamaLLM(model="llama3:8b")
memory = ConversationBufferMemory(input_key="input", memory_key="history")

SESSION_LOCKS: dict = {}   # maps session_id -> {"locked": bool, "station": str, "fare": float, "interchange": str}

CONFIRM_TOKENS_EN = {"yes", "yep", "ok", "okay", "confirm", "buy", "sure"}
CANCEL_TOKENS_EN  = {"no", "nope", "not yet", "later", "cancel", "stop", "back"}

CONFIRM_TOKENS_MS = {"ya", "yao", "ya.", "sah", "betul", "ok", "okey", "oke", "confirm", "boleh", "setuju"}
CANCEL_TOKENS_MS  = {"tidak", "tak", "batal", "jangan", "nanti", "tidaklah", "stop", "balik"}

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
            "station_line1": [],
            "station_line2": [],
            "interchange_station": []
        },
        "query_type": query_type
    }

# detect the language of user input
def detect_language_safely(text: str) -> str:
    MALAY_HINTS = [
        r"\bsaya\b", r"\bnak\b", r"\bpergi\b", r"\bdari\b", r"\bke\b",
        r"\bdengan\b", r"\bkereta\b", r"\bstesen\b", r"\bmrt\b", r"\blrt\b",
        r"\btidak\b", r"\bya\b", r"\bboleh\b", r"\bdi\b"
    ]
    text = text.strip().lower()
    if re.search(r'[\u4e00-\u9fff]', text):
        return 'zh'
    for pattern in MALAY_HINTS:
        if re.search(pattern, text):
            return 'ms'
    try:
        lang = detect(text)
        if lang in ['en', 'zh-cn', 'zh-tw', 'zh']:
            return 'zh' if 'zh' in lang else lang
        elif lang in ['id', 'so', 'ms']:
            return 'ms'
        else:
            return lang
    except:
        return 'en'

# process the user input for better extraction station matching
def preprocess_for_station_matching(text: str, lang: str) -> str:
    text = text.lower()

    if lang == "ms":
        # Remove common Malay filler words
        stopwords = ["saya", "nak", "mahu", "ingin", "pergi", "ke", "stesen", "lrt", "mrt", "tiket"]
        for w in stopwords:
            text = re.sub(rf"\b{w}\b", "", text)
    
    if lang == "en":
        # Remove common English filler words
        stopwords = ["i", "want", "to", "go", "get", "ticket", "station", "lrt", "mrt"]
        for w in stopwords:
            text = re.sub(rf"\b{w}\b", "", text)

    return text.strip()

# generate a random ticket ID
def generate_ticket_id(length=6):
    return ''.join(random.choices(string.ascii_uppercase + string.digits, k=length))

# return a per-session lock dict; create if missing
def get_session_lock(session_id: str):
    if session_id not in SESSION_LOCKS:
        SESSION_LOCKS[session_id] = {"locked": False, "station": None, "fare": None, "interchange": None}
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
    "en": "Okay, I cancelled your ticket request.",
    "ms": "Baik, saya batalkan permintaan tiket anda."
}

# ticket agent prompt
ticket_prompt = PromptTemplate(
    input_variables=["history", "input", "kiosk_station", "fare_info", "destination", "interchange_note", "ticket_status", "language"],
    template="""
<|begin_of_text|>
<|start_header_id|>system<|end_header_id|>
You are a multilingual LRT/MRT ticketing assistant.

Kiosk Station: {kiosk_station}
Conversation so far: {history}

Fare Information: {fare_info}
Destination Station: {destination}
Interchange Information: {interchange_note}
Ticket Status: {ticket_status}

Guidelines:
1. Use the **Fare Information** and **Interchange Information** above exactly. Do NOT invent, change, or recalculate any fares, station names, or ticket IDs.
2. If ticket_status is "no":
   - Mention the ticket details exactly as provided (example: "Your ticket from {kiosk_station} to {destination} costs {fare_info}.")
   - Politely ask the user if they want to confirm this ticket purchase.
   - If the user declines (says no, cancel, not yet, later, stop, back):
     * Politely acknowledge the cancellation (e.g. "Okay, I’ve cancelled your request.").
3. If ticket_status is "yes":
   - Do NOT create ticket IDs or make irreversible actions (the system will handle the actual ticket creation).
   - Only acknowledge: e.g., "Okay, generating your ticket now...".
4. Keep responses short, polite, and always in the user’s language.
IMPORTANT: Always respond in **{language}** and do NOT change any values shown in Fare Information or Interchange Information.
<|eot_id|>
<|start_header_id|>user<|end_header_id|>
{input}
<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>
"""
)

# qna agent prompt
qna_prompt = PromptTemplate(
    input_variables=["history", "input", "kiosk_station", "fare_info", "destination", "interchange_note", "language"],
    template="""
You are a multilingual LRT/MRT information assistant.
Answer user's transport-related questions briefly and politely in **{language}**.
Use the Fare Info and Referred Station below exactly when referencing fares. Just say "don't know" when no information is available.

Current kiosk station: {kiosk_station}
Conversation so far: {history}

Fare Info: {fare_info}
Referred Station: {destination}
Interchange Info: {interchange_note}
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

    # language used for confirmation/cancel detection (based on current message)
    user_lang = detect_language_safely(state["input"])

    # If ticket flow already started (station locked) → confirmation/cancellation handled by ticket_agent
    if lock["station"] is not None:
        if is_confirmation_text(text, user_lang) or is_cancellation_text(text, user_lang):
            state["route"] = "ticket_agent"
            print(f"[ROUTER] Routing to: {state['route']} for input: {state['input']} (confirmation/cancellation) session={session_id} lock={lock}")
            return state

    if any(word in text for word in ["path", "route", "how to get", "how do i go to", "directions", "jalan", "cara pergi"]):
        state["route"] = "route_planning"
    elif any(word in text for word in ["buy", "ticket", "fare", "price", "i want to go", "ke", "pergi"]):
        state["route"] = "ticket_agent"
    else:
        state["route"] = "qna_agent"
        print(f"[ROUTER] Routing to: {state['route']} for input: {state['input']} session={session_id}")
    return state

def ticket_agent_node(state: KioskState) -> KioskState:
    print(f"[AGENT] Running ticket_agent_node for: {state['input']}")
    session_id = state.get("session_id", "default")
    lock = get_session_lock(session_id)

    user_input = state["input"]
    user_lang = detect_language_safely(user_input)
    lang_instruction = "Malay" if user_lang == "ms" else "English"
    history = state.get("history", "")

    fare_info = "None"
    destination = "None"
    interchange_note = "None"

    # Case 1: Waiting for confirmation -> handle confirm/cancel
    if lock["station"] is not None and not lock["locked"]:
        if is_confirmation_text(user_input, user_lang):
            lock["locked"] = True
            print(f"[AGENT] session={session_id} confirmation detected -> locking for station={lock['station']}")
        elif is_cancellation_text(user_input, user_lang):
            SESSION_LOCKS[session_id] = {"locked": False, "station": None, "fare": None, "interchange": None}
            reply_text = CANCEL_MESSAGES.get(user_lang, CANCEL_MESSAGES["en"])
            return_data = {
                "input": user_input,
                "history": history,
                "output": reply_text,
                "route": "",
                "json": build_json_response(
                    text=reply_text,
                    query_type="ticket_agent",
                    session_id=session_id,
                    ticket_details=None,
                    route_details=None
                )
            }
            print(f"### DEBUG [ticket_agent_node] Returning={return_data}")
            return return_data

    # Case 2: Detect new station if none locked
    if lock["station"] is None:
        clean_input = preprocess_for_station_matching(user_input, user_lang)
        station, _, _ = find_station(clean_input)
        if station:
            fare, _, _ = fare_system.get_fare(station)
            interchange_station = fare_system.find_interchange_station(station)
            interchange_note = f"Includes interchange at {interchange_station}." if interchange_station else ""
            fare_info = f"To {station}: RM{fare:.2f}" if fare is not None else ""
            destination = f"{station}"
            if fare is not None:
                lock.update({"station": station, "fare": fare, "interchange": interchange_note})
            print(f"[AGENT] session={session_id} locked station candidate set -> {lock}")
        else:
            fare_info = "None"
    else:
        fare_info = f"To {lock['station']}: RM{lock['fare']:.2f}"
        destination = f"{lock['station']}"
        interchange_note = lock["interchange"]

    # Case 3: If confirmed -> issue ticket
    if lock["locked"]:
        ticket_id = generate_ticket_id()
        current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        ticket_text_en = (
            f"Your ticket is issued! From {kiosk_station} to {lock['station']} "
            f"with a fare of RM{lock['fare']:.2f}. {lock['interchange']}"
            f"Please download this PDF ticket, enjoy your journey!"
        )
        ticket_text_ms = (
            f"Tiket anda sudah dikeluarkan. Dari {kiosk_station} ke {lock['station']} "
            f"dengan tambang RM{lock['fare']:.2f}. {lock['interchange']}"
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

        memory.save_context({"input": "[SYSTEM] Transaction Completed"},
                            {"output": f"--- Ticket transaction completed: {ticket_id} ---"})
        memory.clear()
        SESSION_LOCKS[session_id] = {"locked": False, "station": None, "fare": None, "interchange": None}
        print(f"[AGENT] session={session_id} ticket issued -> {ticket_id} for station={lock['station']}")
        
        return_data = {
            "input": user_input,
            "history": "",
            "output": ticket_text,
            "route": "",
            "json": build_json_response(
                text=ticket_text,
                query_type="ticket_agent",
                session_id=session_id,
                ticket_details=ticket_details,
                route_details=None
            )
        }
        
        print(f"### DEBUG [ticket_agent_node] Returning={return_data}")
        
        return return_data

    # Case 4: Ask for confirmation via LLM
    reply = llm.invoke(ticket_prompt.format(
        history=history,
        input=user_input,
        kiosk_station=kiosk_station,
        fare_info=fare_info,
        destination=destination,
        interchange_note=interchange_note,
        ticket_status="no",
        language=lang_instruction
    ))
    print(f"[AGENT] session={session_id} sent confirm prompt (language={lang_instruction}) lock={lock}")

    return_data = {
        "input": user_input,
        "history": history,
        "output": reply,
        "route": "ticket_agent",
        "json": build_json_response(
            text=reply,
            query_type="ticket_agent",
            session_id=session_id,
            ticket_details=None,
            route_details=None
        )
    }
    print(f"### DEBUG [ticket_agent_node] Returning={return_data}")
    return return_data

def qna_agent_node(state: KioskState) -> KioskState:
    print(f"[AGENT] Running qna_agent_node for: {state['input']}")
    user_input = state["input"]
    user_lang = detect_language_safely(user_input)
    lang_instruction = "Malay" if user_lang == "ms" else "English"
    history = state.get("history", "")

    fare_info = "None"
    interchange_note = "None"
    destination = "None"
    clean_input = preprocess_for_station_matching(user_input, user_lang)
    station, _, _ = find_station(clean_input)
    if station:
        fare, _, _ = fare_system.get_fare(station)
        interchange_station = fare_system.find_interchange_station(station)
        interchange_note = f"Includes interchange at {interchange_station}" if interchange_station else ""
        fare_info = f"To {station}: RM{fare:.2f}" if fare is not None else ""
        destination = f"{station}"

    reply = llm.invoke(qna_prompt.format(
        history=history,
        input=user_input,
        kiosk_station=kiosk_station,
        fare_info=fare_info,
        destination=destination,
        interchange_note=interchange_note,
        language=lang_instruction
    ))

    memory.clear()
    
    return_data = {
        "input": user_input,
        "history": history,
        "output": reply,
        "route": "",
        "json": build_json_response(
            text=reply,
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
    user_input = state["input"]
    user_lang = detect_language_safely(user_input)
    history = state.get("history", "")
    
    clean_input = preprocess_for_station_matching(user_input, user_lang)
    dest_station, _, _ = find_station(clean_input)
    
    if not dest_station:
        reply_en = "I couldn’t find the destination station. Please try again."
        reply_ms = "Saya tidak dapat mencari stesen destinasi. Sila cuba lagi."
        reply_text = reply_en if user_lang != "ms" else reply_ms
        seg1, seg2, interchange = [], [], None
    else:
        route_text_en, route_text_ms, seg1, seg2, interchange = plan_route(kiosk_station, dest_station, fare_system)
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
                "station_line1": seg1,
                "station_line2": seg2,
                "interchange_station": [interchange] if interchange else []
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
    state = {
        "input": user_message,
        "history": memory.load_memory_variables({}).get("history", ""),
        "output": "",
        "route": "",
        "session_id": session_id
    }
    print(f"\n### DEBUG: Incoming state={state}")
    
    result = app.invoke(
        state,
        config={"configurable": {"thread_id": session_id}}
    )
    
    print(f"### DEBUG: Raw result from workflow={result}")

    # only persist if it's ticket agent
    if result["route"] == "ticket_agent":
        memory.save_context({"input": user_message}, {"output": result["output"]})
        
    # fallback JSON
    json_data = result.get("json")
    if not json_data:  
        json_data = build_json_response(
            text=result.get("output", ""),
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
        "station_line1": [],
        "station_line2": [],
        "interchange_station": []
    })
    
    print(f"### DEBUG: Final JSON response={json_data}\n")

    return json_data
