# huggingface_llm.py
from typing import TypedDict
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

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# ==== INIT & GLOBALS ====
kiosk_station = "Kelana Jaya"
fare_system = FareSystem("Fare.csv", from_station=kiosk_station)

memory = ConversationBufferMemory(input_key="input", memory_key="history")
SESSION_LOCKS: dict = {}  # maps session_id -> {"locked": bool, "station": str, "fare": float, "interchange": str}

CONFIRM_TOKENS_EN = {"yes", "yep", "ok", "okay", "confirm", "buy", "sure"}
CANCEL_TOKENS_EN  = {"no", "nope", "not yet", "later", "cancel", "stop", "back"}
CONFIRM_TOKENS_MS = {"ya", "yao", "ya.", "sah", "betul", "ok", "okey", "oke", "confirm", "boleh", "setuju"}
CANCEL_TOKENS_MS  = {"tidak", "tak", "batal", "jangan", "nanti", "tidaklah", "stop", "balik"}

CANCEL_MESSAGES = {
    "en": "Okay, I cancelled your ticket request.",
    "ms": "Baik, saya batalkan permintaan tiket anda."
}

# ==== HUGGINGFACE LLM SETUP ====
device = 0 if torch.cuda.is_available() else -1
model_name = "meta-llama/Meta-Llama-3-8B"
token = "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"
tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=token)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    torch_dtype=torch.float16,
    load_in_8bit=True  # Reduce VRAM usage
)

llm_pipeline = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=512,
    device=device,
    do_sample=True,
    temperature=0.7
)

def invoke_llm(prompt: str) -> str:
    output = llm_pipeline(prompt)
    return output[0]["generated_text"]

# ==== HELPERS ====
def build_json_response(text: str, query_type: str, session_id: str, ticket_details: dict = None, route_details: dict = None):
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

def preprocess_for_station_matching(text: str, lang: str) -> str:
    text = text.lower()
    stopwords = ["saya", "nak", "mahu", "ingin", "pergi", "ke", "stesen", "lrt", "mrt", "tiket"] if lang=="ms" else ["i", "want", "to", "go", "get", "ticket", "station", "lrt", "mrt"]
    for w in stopwords:
        text = re.sub(rf"\b{w}\b", "", text)
    return text.strip()

def generate_ticket_id(length=6):
    return ''.join(random.choices(string.ascii_uppercase + string.digits, k=length))

def get_session_lock(session_id: str):
    if session_id not in SESSION_LOCKS:
        SESSION_LOCKS[session_id] = {"locked": False, "station": None, "fare": None, "interchange": None}
    return SESSION_LOCKS[session_id]

def is_confirmation_text(text: str, lang: str = "en") -> bool:
    tokens = set(re.findall(r"\w+", (text or "").lower()))
    return bool(tokens & (CONFIRM_TOKENS_MS if lang=="ms" else CONFIRM_TOKENS_EN))

def is_cancellation_text(text: str, lang: str = "en") -> bool:
    tokens = set(re.findall(r"\w+", (text or "").lower()))
    return bool(tokens & (CANCEL_TOKENS_MS if lang=="ms" else CANCEL_TOKENS_EN))

# ==== PROMPTS ====
ticket_prompt = PromptTemplate(
    input_variables=["history", "input", "kiosk_station", "fare_info", "interchange_note", "ticket_status", "language"],
    template="""
You are a multilingual LRT/MRT ticketing assistant.

Kiosk Station: {kiosk_station}
Conversation so far: {history}

Fare Information: {fare_info}
Interchange Information: {interchange_note}
Ticket Status: {ticket_status}

Guidelines:
1. Use the Fare Information exactly.
2. Keep responses short, polite, and in {language}.
User: {input}
Assistant:"""
)

qna_prompt = PromptTemplate(
    input_variables=["history", "input", "kiosk_station", "fare_info", "interchange_note", "language"],
    template="""
You are a multilingual LRT/MRT information assistant.
Answer user's transport-related questions briefly in {language}.

Current kiosk station: {kiosk_station}
Conversation so far: {history}

Fare Info: {fare_info}
Interchange Info: {interchange_note}
Question: {input}
"""
)

# ==== STATE TYPE ====
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

    user_lang = detect_language_safely(state["input"])

    if lock["station"] is not None:
        if is_confirmation_text(text, user_lang) or is_cancellation_text(text, user_lang):
            state["route"] = "ticket_agent"
            return state

    if any(word in text for word in ["path", "route", "how to get", "how do i go to", "directions", "jalan", "cara pergi"]):
        state["route"] = "route_planning"
    elif any(word in text for word in ["buy", "ticket", "fare", "price", "i want to go", "ke", "pergi"]):
        state["route"] = "ticket_agent"
    else:
        state["route"] = "qna_agent"
    return state

def ticket_agent_node(state: KioskState) -> KioskState:
    session_id = state.get("session_id", "default")
    lock = get_session_lock(session_id)
    user_input = state["input"]
    user_lang = detect_language_safely(user_input)
    lang_instruction = "Malay" if user_lang == "ms" else "English"
    history = state.get("history", "")

    fare_info, interchange_note = "None", "None"

    if lock["station"] is not None and not lock["locked"]:
        if is_confirmation_text(user_input, user_lang):
            lock["locked"] = True
        elif is_cancellation_text(user_input, user_lang):
            SESSION_LOCKS[session_id] = {"locked": False, "station": None, "fare": None, "interchange": None}
            reply_text = CANCEL_MESSAGES.get(user_lang, CANCEL_MESSAGES["en"])
            return {
                "input": user_input,
                "history": history,
                "output": reply_text,
                "route": "",
                "json": build_json_response(reply_text, "ticket_agent", session_id)
            }

    if lock["station"] is None:
        clean_input = preprocess_for_station_matching(user_input, user_lang)
        station, _, _ = find_station(clean_input)
        if station:
            fare, _, _ = fare_system.get_fare(station)
            interchange_station = fare_system.find_interchange_station(station)
            interchange_note = f"Includes interchange at {interchange_station}" if interchange_station else "None"
            fare_info = f"To {station}: RM{fare:.2f}" if fare is not None else "None"
            if fare is not None:
                lock.update({"station": station, "fare": fare, "interchange": interchange_note})
    else:
        fare_info = f"To {lock['station']}: RM{lock['fare']:.2f}"
        interchange_note = lock["interchange"]

    if lock["locked"]:
        ticket_id = generate_ticket_id()
        current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        ticket_text = (
            f"Ticket ID {ticket_id} issued on {current_time}, from {kiosk_station} to {lock['station']} "
            f"with a fare of RM{lock['fare']:.2f}. {lock['interchange']}."
        )
        ticket_details = {
            "session_id": session_id,
            "ticket_id": ticket_id,
            "from_station": kiosk_station,
            "to_station": lock["station"],
            "fare": f"RM{lock['fare']:.2f}",
            "interchange": lock["interchange"],
            "datetime": current_time
        }
        memory.clear()
        SESSION_LOCKS[session_id] = {"locked": False, "station": None, "fare": None, "interchange": None}
        return {
            "input": user_input,
            "history": "",
            "output": ticket_text,
            "route": "",
            "json": build_json_response(ticket_text, "ticket_agent", session_id, ticket_details)
        }

    # Ask for confirmation via LLM
    reply = invoke_llm(ticket_prompt.format(
        history=history,
        input=user_input,
        kiosk_station=kiosk_station,
        fare_info=fare_info,
        interchange_note=interchange_note,
        ticket_status="no",
        language=lang_instruction
    ))

    return {
        "input": user_input,
        "history": history,
        "output": reply,
        "route": "ticket_agent",
        "json": build_json_response(reply, "ticket_agent", session_id)
    }

def qna_agent_node(state: KioskState) -> KioskState:
    user_input = state["input"]
    user_lang = detect_language_safely(user_input)
    lang_instruction = "Malay" if user_lang == "ms" else "English"
    history = state.get("history", "")
    
    clean_input = preprocess_for_station_matching(user_input, user_lang)
    station, _, _ = find_station(clean_input)
    fare_info, interchange_note = "None", "None"
    if station:
        fare, _, _ = fare_system.get_fare(station)
        interchange_station = fare_system.find_interchange_station(station)
        interchange_note = f"Includes interchange at {interchange_station}" if interchange_station else "None"
        fare_info = f"To {station}: RM{fare:.2f}" if fare is not None else "None"

    reply = invoke_llm(qna_prompt.format(
        history=history,
        input=user_input,
        kiosk_station=kiosk_station,
        fare_info=fare_info,
        interchange_note=interchange_note,
        language=lang_instruction
    ))

    memory.clear()
    return {
        "input": user_input,
        "history": history,
        "output": reply,
        "route": "",
        "json": build_json_response(reply, "qna_agent", state["session_id"])
    }

def route_planning_node(state: KioskState) -> KioskState:
    user_input = state["input"]
    user_lang = detect_language_safely(user_input)
    history = state.get("history", "")

    clean_input = preprocess_for_station_matching(user_input, user_lang)
    dest_station, _, _ = find_station(clean_input)
    if not dest_station:
        reply_text = "Saya tidak dapat mencari stesen destinasi. Sila cuba lagi." if user_lang=="ms" else "I couldn’t find the destination station. Please try again."
        seg1, seg2, interchange = [], [], None
    else:
        route_text_en, route_text_ms, seg1, seg2, interchange = plan_route(kiosk_station, dest_station, fare_system)
        reply_text = route_text_ms if user_lang=="ms" else route_text_en

    memory.clear()
    return {
        "input": user_input,
        "history": history,
        "output": reply_text,
        "route": "",
        "json": build_json_response(reply_text, "route_planning", state["session_id"], route_details={
            "station_line1": seg1,
            "station_line2": seg2,
            "interchange_station": [interchange] if interchange else []
        })
    }

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

    result = app.invoke(state, config={"configurable": {"thread_id": session_id}})
    if result["route"] == "ticket_agent":
        memory.save_context({"input": user_message}, {"output": result["output"]})

    json_data = result.get("json") or build_json_response(result.get("output", ""), result.get("route", "unknown"), session_id)
    json_data.setdefault("ticket_details", {"session_id": "", "ticket_id": "", "from_station": "", "to_station": "", "fare": "", "interchange": "", "datetime": ""})
    json_data.setdefault("route_details", {"station_line1": [], "station_line2": [], "interchange_station": []})

    return json_data
