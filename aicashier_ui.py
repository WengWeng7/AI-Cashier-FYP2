# --- ASR Dependencies ---
import tempfile
from streamlit_realtime_audio_recorder import audio_recorder
import base64
from pydub import AudioSegment
# --- LLM Dependencies ---
import streamlit as st
import uuid
import time
import requests
import io
import qrcode
from typing import Dict, List
import pydeck as pdk
import pandas as pd
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib import colors
from reportlab.lib.units import mm
from station_coords import station_coords
# --- TTS Dependencies ---
import winsound
import threading

ASR_URL = "http://localhost:8000/transcribe"
ROOT_URL = "http://localhost:8010/"
LLM_URL = "http://localhost:8010/chat"
TTS_URL = "http://localhost:8020/infer" 

st.set_page_config(page_title="AI Ticketing Kiosk", page_icon="🚉", layout="wide")
st.title("🚉 AI Ticketing Kiosk Assistant")

# ---------------------------
# Session bootstrap
# ---------------------------
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

if "messages" not in st.session_state:
    try:
        resp = requests.get(ROOT_URL).json()
        welcome_msg = resp.get("message", "Welcome to AI Ticketing Kiosk!")
    except Exception:
        welcome_msg = "Welcome to AI Ticketing Kiosk!"
    st.session_state.messages = [{"role": "assistant", "content": welcome_msg}]

# maintain last ai reply for sidebar JSON display
if "last_ai_reply" not in st.session_state:
    st.session_state.last_ai_reply = None
    
# ---------------------------
# ASR UI Helpers
# ---------------------------
def process_audio(audio_bytes):
    """Convert WebM audio bytes to WAV and save to temp file"""
    try:
        # Decode webm bytes with pydub
        audio_io = io.BytesIO(audio_bytes)
        audio = AudioSegment.from_file(audio_io, format="webm")

        # Ensure mono + 16kHz for ASR
        audio = audio.set_channels(1).set_frame_rate(16000).set_sample_width(2)

        # Save as temporary WAV
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            audio.export(tmp.name, format="wav",  codec="pcm_s16le")
            return tmp.name

    except Exception as e:
        st.error(f"Error processing audio: {str(e)}")
        return None

def transcribe_audio(file_path):
    """Send audio file to API for transcription"""
    try:
        with open(file_path, 'rb') as f:
            response = requests.post(ASR_URL, files={"file": ("speech.wav", f, "audio/wav")})
            
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"API Error: {response.text}")
            return None
    except Exception as e:
        st.error(f"Error during transcription: {str(e)}")
        return None


# ---------------------------
# LLM UI Helpers
# ---------------------------
def get_ai_reply(user_input: str, session_id: str) -> Dict:
    try:
        with st.spinner("Loading..."):
            time.sleep(0.4)
            payload = {"user_message": user_input, "session_id": session_id}
            resp = requests.post(LLM_URL, json=payload, timeout=30).json()
            return resp
    except Exception as e:
        st.error(f"Sorry, there was an error: {str(e)}")
        return {
            "text": "⚠️ Error encountered. Please try again.",
            "ticket_details": {},
            "route_details": {},
            "query_type": "error",
        }


def generate_ticket_pdf(ticket: Dict) -> bytes:
    """
    Creates a well-sized rectangular ticket on a Letter page and returns PDF bytes.
    """
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=letter)
    width, height = letter

    # Ticket card dimensions (approx. 160mm x 70mm)
    card_w = 160 * mm
    card_h = 70 * mm
    margin_x = (width - card_w) / 2
    margin_y = height - (card_h + 60)

    # Draw card background
    c.setFillColor(colors.whitesmoke)
    c.roundRect(margin_x, margin_y, card_w, card_h, 8 * mm, fill=1, stroke=0)

    # Border
    c.setLineWidth(1)
    c.setStrokeColor(colors.grey)
    c.roundRect(margin_x, margin_y, card_w, card_h, 8 * mm, fill=0, stroke=1)

    # Header bar
    c.setFillColor(colors.lightblue)
    c.roundRect(margin_x, margin_y + card_h - 18 * mm, card_w, 18 * mm, 8 * mm, fill=1, stroke=0)
    c.setFillColor(colors.black)
    c.setFont("Helvetica-Bold", 16)
    c.drawString(margin_x + 12 * mm, margin_y + card_h - 12 * mm, "RapidKL AI Assistant Ticket")

    # Ticket fields
    c.setFont("Helvetica-Bold", 12)
    y = margin_y + card_h - 26 * mm
    line_gap = 7 * mm

    def field(label, value):
        nonlocal y
        c.setFont("Helvetica-Bold", 11)
        c.drawString(margin_x + 12 * mm, y, f"{label}:")
        c.setFont("Helvetica", 11)
        c.drawString(margin_x + 45 * mm, y, str(value or "-"))
        y -= line_gap

    field("Ticket ID", ticket.get("ticket_id", ""))
    field("Session ID", ticket.get("session_id", ""))
    field("From", ticket.get("from_station", ""))
    field("To", ticket.get("to_station", ""))
    field("Fare", ticket.get("fare", ""))
    field("Interchange", ticket.get("interchange", ""))
    field("Date/Time", ticket.get("datetime", ""))

    # Footer note
    c.setFont("Helvetica-Oblique", 9)
    c.setFillColor(colors.grey)
    c.drawRightString(margin_x + card_w - 10 * mm, margin_y + 6 * mm, "Enjoy your journey!")

    c.showPage()
    c.save()
    buf.seek(0)
    return buf.read()


def render_ticket_preview(ticket: Dict):
    if not ticket or not ticket.get("ticket_id"):
        return  

    st.markdown("#### 🎟 Ticket Preview")
    st.markdown(
        f"""
        <div style="
            border:1px solid #ddd;
            border-radius:16px;
            padding:16px 18px;
            background:linear-gradient(180deg, #ffffff, #f8fbff);
            box-shadow:0 4px 14px rgba(0,0,0,0.06);
            max-width:640px;">
          <div style="display:flex;align-items:center;margin-bottom:8px;">
            <div style="font-weight:700;font-size:18px;">AI Ticketing Kiosk</div>
          </div>
          <div style="display:grid;grid-template-columns:130px 1fr;row-gap:8px;column-gap:10px;font-size:14px;">
            <div><b>Ticket ID</b></div><div>{ticket.get("ticket_id","")}</div>
            <div><b>Session ID</b></div><div style="word-break:break-all;">{ticket.get("session_id","")}</div>
            <div><b>From</b></div><div>{ticket.get("from_station","")}</div>
            <div><b>To</b></div><div>{ticket.get("to_station","")}</div>
            <div><b>Fare</b></div><div>{ticket.get("fare","")}</div>
            <div><b>Interchange</b></div><div>{ticket.get("interchange","")}</div>
            <div><b>Date/Time</b></div><div>{ticket.get("datetime","")}</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    
    # Generate PDF bytes
    pdf_bytes = generate_ticket_pdf(ticket)
    
    st.download_button(
        "⬇️ Download PDF Ticket",
        data=pdf_bytes,
        file_name=f"ticket_{ticket.get('ticket_id', uuid.uuid4())}.pdf",
        mime="application/pdf",
        key=f"download_{ticket.get('ticket_id', uuid.uuid4())}",
        use_container_width=True,
    )


def build_route_text(station_lines: List[List[str]], interchanges: List[str]) -> str:
    """
    Builds a fancy arrow route string with multiple interchanges.
    Shows each interchange only once (via ⏩ markers).
    """
    def chain(arr: List[str]) -> str:
        return " ➡️ ".join(arr) if arr else ""

    parts = []
    for idx, seg in enumerate(station_lines):
        if not seg:
            continue

        if idx == 0:
            # First segment: keep as is
            parts.append(chain(seg))
        else:
            # Interchange station for this transition
            inter = interchanges[idx-1] if idx-1 < len(interchanges) else None
            if inter:
                # Remove interchange from previous segment’s end
                if parts and parts[-1].endswith(inter):
                    parts[-1] = parts[-1].rsplit(" ➡️ " + inter, 1)[0]

                # Remove interchange from next segment’s start
                seg = seg[1:] if seg and seg[0] == inter else seg

                # Insert interchange marker
                parts.append(f" ⏩ {inter} ⏩ ")

            # Append cleaned segment
            if seg:
                parts.append(chain(seg))

    return "".join(parts)



def render_route_map(route_details: Dict, station_coords: Dict[str, tuple], delay: float = 0.8):
    """
    Render the route map for multiple station lines and interchanges.
    """
    if not route_details or not route_details.get("station_lines"):
        return
    
    station_lines = route_details.get("station_lines", []) or []
    interchanges = route_details.get("interchanges", []) or []

    # Build full ordered list of stations with interchanges in between
    full_names = []
    for idx, seg in enumerate(station_lines):
        if idx > 0 and idx-1 < len(interchanges):
            inter = interchanges[idx-1]
            if full_names and full_names[-1] != inter:
                full_names.append(inter)
        full_names.extend(seg)

    # Ensure we have coords for each station
    coords = []
    names_with_coords = []
    for name in full_names:
        pt = station_coords.get(name)
        if pt:
            lat, lon = pt
            coords.append([lon, lat])  # pydeck expects [lon, lat]
            names_with_coords.append(name)
        else:
            st.warning(f"Missing coords for station: {name} — skipped on map.")
    
    if len(coords) < 2:
        st.warning("Not enough station coordinates to draw a route.")
        return

    # DataFrame for all stations
    df_points = pd.DataFrame([
        {"station": n, "lat": station_coords[n][0], "lon": station_coords[n][1]}
        for n in names_with_coords
    ])

    # Tile layer: OpenStreetMap (no API key)
    tile_layer = pdk.Layer(
        "TileLayer",
        data="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png",
        min_zoom=0,
        max_zoom=19,
        tile_size=256,
        opacity=1.0
    )

    # Generic station markers
    station_layer = pdk.Layer(
        "ScatterplotLayer",
        data=df_points,
        get_position='[lon, lat]',
        get_radius=100,
        get_fill_color=[200, 30, 30],
        pickable=True,
        auto_highlight=True
    )

    # Special markers: start (green), end (red), interchanges (orange)
    special_markers = []

    if names_with_coords:
        sname = names_with_coords[0]
        slat, slon = station_coords[sname]
        special_markers.append({"pos": [slon, slat], "color": [34, 139, 34], "label": "Start", "name": sname})

        ename = names_with_coords[-1]
        elat, elon = station_coords[ename]
        special_markers.append({"pos": [elon, elat], "color": [200, 30, 30], "label": "End", "name": ename})

    for inter in interchanges:
        if inter in station_coords:
            ilat, ilon = station_coords[inter]
            special_markers.append({"pos": [ilon, ilat], "color": [255, 165, 0], "label": "Interchange", "name": inter})

    special_df = pd.DataFrame([
        {"lon": m["pos"][0], "lat": m["pos"][1], "color": m["color"], "label": m["label"], "name": m["name"]}
        for m in special_markers
    ])

    special_layer = pdk.Layer(
        "ScatterplotLayer",
        data=special_df,
        get_position='[lon, lat]',
        get_fill_color='color',
        get_radius=180,
        pickable=True,
    )

    # Placeholder for animation
    placeholder = st.empty()

    # Initial view
    view_state = pdk.ViewState(latitude=coords[0][1], longitude=coords[0][0], zoom=11, pitch=0)

    # Animate path drawing
    for i in range(1, len(coords)):
        completed_path = [{"path": coords[:i+1]}]
        completed_layer = pdk.Layer(
            "PathLayer",
            data=completed_path,
            get_path="path",
            get_color=[0, 128, 255],
            width_scale=6,
            width_min_pixels=4
        )

        current_seg = [{"path": coords[max(0, i-1):i+1]}]
        current_layer = pdk.Layer(
            "PathLayer",
            data=current_seg,
            get_path="path",
            get_color=[255, 165, 0],  # orange
            width_scale=10,
            width_min_pixels=6
        )

        deck = pdk.Deck(
            layers=[tile_layer, completed_layer, current_layer, station_layer, special_layer],
            initial_view_state=view_state,
            tooltip={"html": "<b>Station:</b> {station}", "style": {"backgroundColor": "white", "color": "black"}}
        )
        placeholder.pydeck_chart(deck)
        time.sleep(delay)

    # Final frame
    final_layer = pdk.Layer(
        "PathLayer",
        data=[{"path": coords}],
        get_path="path",
        get_color=[0, 128, 255],
        width_scale=6,
        width_min_pixels=4
    )
    final_deck = pdk.Deck(
        layers=[tile_layer, final_layer, station_layer, special_layer],
        initial_view_state=view_state,
        tooltip={"html": "<b>Station:</b> {station}", "style": {"backgroundColor": "white", "color": "black"}}
    )
    placeholder.pydeck_chart(final_deck)


def render_route_preview(route: Dict):
    if not route or not route.get("station_lines"):
        return 
    
    station_lines = route.get("station_lines", []) or []
    interchanges = route.get("interchanges", []) or []

    route_text = build_route_text(station_lines, interchanges)

    if not route_text:
        st.info("No route segments available.")
        return

    return st.markdown(
        f"""
        <div style="border:1px dashed #cfd6e4; border-radius:14px; padding:14px 16px;
                    background:#fbfdff; box-shadow:0 2px 10px rgba(0,0,0,0.04); max-width:1050px;">
            <div style="font-weight:600;margin-bottom:8px;">Route</div>
            <div style="font-size:15px;">{route_text}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    
# ---------------------------
# TTS UI Helpers
# ---------------------------
def speak_text(text, speaker="Shafiqah Idayu"):
    resp_ms = requests.post("http://localhost:8020/infer", json={
        "text": text,
        "speaker": speaker
    })
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(resp_ms.content)
        tmp_path = tmp.name
    return tmp_path

def play_tts(audio_path):
    winsound.PlaySound(audio_path, winsound.SND_FILENAME)
    
# ---------------------------
# Render chat history FIRST (above the input)
# ---------------------------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        if msg["role"] == "assistant" and msg.get("query_type"):
            st.caption(f"Query Type: {msg['query_type']}")
            
# ---------------------------
# Sidebar: audio recorder + inspector
# ---------------------------
with st.sidebar:
    st.subheader("🎤 Voice Input")
    result = audio_recorder(
        interval=50,
        threshold=-60,
        silenceTimeout=200
    )
    
    # Make sure session state is initialized
    if "transcribed_input" not in st.session_state:
        st.session_state.transcribed_input = None

    # Handle audio recording
    if result:
        if result.get('status') == 'stopped':
            audio_data = result.get('audioData')
            if audio_data:
                audio_bytes = base64.b64decode(audio_data)
                audio_file = process_audio(audio_bytes)
                if audio_file:
                    asr_result = transcribe_audio(audio_file)
                    if asr_result:
                        st.session_state.transcribed_input = asr_result["transcription"]
            else:
                st.warning("No audio data was recorded")
        elif result.get('error'):
            st.error(f"Error: {result.get('error')}")

    st.subheader("ℹ️ Response Inspector")
    last = st.session_state.last_ai_reply
    if last:
        if last.get("ticket_details"):
            with st.expander("🎟 Ticket Details"):
                st.json(last["ticket_details"])
        if last.get("route_details"):
            with st.expander("🗺 Route Details"):
                st.json(last["route_details"])
        if last.get("query_type"):
            st.caption(f"Query Type: {last['query_type']}")
    else:
        st.caption("Interact to see raw details here.")

# ---------------------------
# Chat input at the very bottom (sticky)
# ---------------------------
typed_input = st.chat_input("Type your message here...")

user_input = None

# Priority 1: typed input (if you type, it overrides voice)
if typed_input:
    user_input = typed_input
    st.session_state.transcribed_input = None  # clear voice buffer

# Priority 2: voice input (only if nothing typed)
elif st.session_state.transcribed_input:
    user_input = st.session_state.transcribed_input
    st.session_state.transcribed_input = None  # clear after use

# ---------------------------
# Chat logic - Process new input
# ---------------------------
if user_input:
    # Store assistant response in history (replace old one if any)
    #if st.session_state.messages and st.session_state.messages[-1]["role"] == "assistant":
    #    st.session_state.messages.pop()  # remove previous assistant msg
    
    # Add user message to history
    st.session_state.messages.append({"role": "user", "content": user_input})
    st.chat_message("user").write(user_input)

    # Get AI response
    ai_reply = get_ai_reply(user_input, st.session_state.session_id)
    st.session_state.last_ai_reply = ai_reply

    # Store assistant response in history
    st.session_state.messages.append({
        "role": "assistant",
        "content": ai_reply.get("text", ""),
        "query_type": ai_reply.get("query_type", ""),
        "ticket_details": ai_reply.get("ticket_details", ""),
        "route_details": ai_reply.get("route_details", ""),
    })
    
    with st.chat_message("assistant"):
        # Always show the text immediately
        ai_text = ai_reply.get("text", "")
        
        if ai_text:
            # generate audio first
            with st.spinner("Loading..."):
                audio_path = speak_text(ai_text)
            
            # Create a placeholder for animated text
            placeholder = st.empty()
            typed_text = ""

            threading.Thread(target=play_tts, args=(audio_path,), daemon=True).start()
            for word in ai_text.split():
                typed_text += word + " "
                placeholder.markdown(typed_text)
                time.sleep(0.2)  # typing speed
                
        # Show query type if available
        if ai_reply.get("query_type"):
            st.caption(f"Query Type: {ai_reply['query_type']}")

        # Handle special assistant message features
        if ai_reply.get("ticket_details"):
            render_ticket_preview(ai_reply["ticket_details"])

        if ai_reply.get("route_details"):
            render_route_map(ai_reply["route_details"], station_coords, delay=0.2)
            render_route_preview(ai_reply["route_details"])
