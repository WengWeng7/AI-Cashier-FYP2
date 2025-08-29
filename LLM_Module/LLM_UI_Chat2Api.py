import streamlit as st
import uuid
import time
import requests
import io
import base64
import qrcode
from typing import Dict, List
import pydeck as pdk
import pandas as pd
import os
from station_coords import station_coords

# --- PDF generation (ReportLab) ---
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib import colors
from reportlab.lib.units import mm

API_URL = "http://localhost:8000/chat"
ROOT_URL = "http://localhost:8000/"
QR_URL = "http://localhost:8000/download/{token}" 

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
# Helpers
# ---------------------------
def get_ai_reply(user_input: str, session_id: str) -> Dict:
    try:
        with st.spinner("Kiosk is typing..."):
            time.sleep(0.4)
            payload = {"user_message": user_input, "session_id": session_id}
            resp = requests.post(API_URL, json=payload, timeout=30).json()
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
    
    try:
        with st.spinner("Please wait for awhile. Generating QR ticket for you..."):
            time.sleep(0.4)
            # Generate PDF bytes
            pdf_bytes = generate_ticket_pdf(ticket)

            # Create unique token and store PDF in memory (server)
            token = str(uuid.uuid4())
            from llm_api import pdf_storage  # import the dict from your API
            pdf_storage[token] = pdf_bytes

            # Generate download URL
            download_url = QR_URL.format(token=token)

            # Generate QR code pointing to server URL
            qr = qrcode.QRCode(box_size=6, border=2, error_correction=qrcode.constants.ERROR_CORRECT_L)
            qr.add_data(download_url)
            qr.make(fit=True)
            img = qr.make_image(fill_color="black", back_color="white")
            
            # Convert PIL.Image to bytes for Streamlit
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            buf.seek(0)

            st.image(buf, caption="Scan QR to download ticket PDF", use_container_width=False)
    except Exception as e:
        st.error(f"Sorry, there was an error: {str(e)}")


def build_route_text(seg1: List[str], seg2: List[str], interchange: List[str]) -> str:
    """
    Builds a fancy arrow route string without duplicating interchange stations.
    """
    def chain(arr: List[str]) -> str:
        return " ➡️ ".join(arr) if arr else ""

    parts = []
    if seg1:
        parts.append(chain(seg1))

    if interchange:
        inter = interchange[0]

        # if seg1 ends with interchange, remove duplicate
        if seg1 and seg1[-1] == inter:
            parts[-1] = chain(seg1[:-1]) + f" ➡️ {inter}"

        # always show interchange marker
        parts.append(f" ⏩ {inter} ⏩ ")

        # if seg2 starts with interchange, skip duplicate
        if seg2 and seg2[0] == inter:
            seg2 = seg2[1:]

    if seg2:
        parts.append(chain(seg2))

    return "".join(parts)


def render_route_map(route_details: Dict, station_coords: Dict[str, tuple], delay: float = 0.8):
    if not route_details or not route_details.get("station_line1"):
        return
    seg1 = route_details.get("station_line1", []) or []
    seg2 = route_details.get("station_line2", []) or []
    interchange = route_details.get("interchange", []) or []

    # build full path (avoid repeating interchange if seg2 starts with the same)
    if seg2 and interchange and seg2[0] == interchange[0]:
        full_names = seg1 + seg2[1:]  # skip duplicate interchange
    elif seg2:
        full_names = seg1 + seg2
    else:
        full_names = seg1

    # fallback: if seg2 empty just use seg1
    if not seg2:
        full_names = seg1

    # ensure we have coords for each station; skip stations without coords
    coords = []
    names_with_coords = []
    for name in full_names:
        pt = station_coords.get(name)
        if pt:
            lat, lon = pt  # station_coords expected (lat, lon)
            coords.append([lon, lat])  # pydeck wants [lon, lat]
            names_with_coords.append(name)
        else:
            st.warning(f"Missing coords for station: {name} — it will be skipped on the map.")
    
    if len(coords) < 2:
        st.warning("Not enough station coordinates to draw a route.")
        return

    # DataFrame for station markers (use lon,lat columns for pydeck scatter)
    df_points = pd.DataFrame([{"station": n, "lat": station_coords[n][0], "lon": station_coords[n][1]}
                              for n in names_with_coords])

    # Tile layer: OpenStreetMap tiles (no API key)
    tile_layer = pdk.Layer(
        "TileLayer",
        data="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png",
        min_zoom=0,
        max_zoom=19,
        tile_size=256,
        opacity=1.0
    )

    # Station markers; we will show all at once
    station_layer = pdk.Layer(
        "ScatterplotLayer",
        data=df_points,
        get_position='[lon, lat]',
        get_radius=100,
        get_fill_color=[200, 30, 30],
        pickable=True,
        auto_highlight=True
    )

    # highlight start/end/interchange indices (if available)
    start_idx = 0
    end_idx = len(names_with_coords) - 1
    inter_name = interchange[0] if interchange else None

    # placeholders for custom marker layers (start/end/interchange)
    special_markers = []
    # start marker (green)
    if names_with_coords:
        sname = names_with_coords[0]
        slat, slon = station_coords[sname]
        special_markers.append({"pos": [slon, slat], "color": [34, 139, 34], "label": "Start", "name": sname})
    # end marker (red)
    if names_with_coords:
        ename = names_with_coords[-1]
        elat, elon = station_coords[ename]
        special_markers.append({"pos": [elon, elat], "color": [200, 30, 30], "label": "End", "name": ename})
    # interchange (orange) — if present and has coords
    if inter_name and inter_name in station_coords:
        ilat, ilon = station_coords[inter_name]
        special_markers.append({"pos": [ilon, ilat], "color": [255, 165, 0], "label": "Interchange", "name": inter_name})

    # a scatter layer for special markers
    special_df = pd.DataFrame([{"lon": m["pos"][0], "lat": m["pos"][1], "color": m["color"], "label": m["label"], "name": m["name"]} for m in special_markers])
    special_layer = pdk.Layer(
        "ScatterplotLayer",
        data=special_df,
        get_position='[lon, lat]',
        get_fill_color='color',
        get_radius=180,
        pickable=True,
    )

    # progress placeholder
    placeholder = st.empty()

    # initial view centered on first station
    view_state = pdk.ViewState(latitude=coords[0][1], longitude=coords[0][0], zoom=11, pitch=0)

    # progressive drawing: each iteration we show the partial path coords[:i+1]
    for i in range(1, len(coords)):
        # path up to i (completed)
        completed_path = [{"path": coords[:i+1]}]  # one path with coordinates so far

        completed_layer = pdk.Layer(
            "PathLayer",
            data=completed_path,
            get_path="path",
            get_color=[0, 128, 255],
            width_scale=6,
            width_min_pixels=4
        )

        # optionally highlight the "current" segment
        current_seg = [{"path": coords[max(0, i-1):i+1]}]  # last edge (two points)
        current_layer = pdk.Layer(
            "PathLayer",
            data=current_seg,
            get_path="path",
            get_color=[255, 165, 0],  # orange for the segment being drawn
            width_scale=10,
            width_min_pixels=6
        )

        # assemble layers: tile, completed, current, points, special markers
        deck = pdk.Deck(
            layers=[tile_layer, completed_layer, current_layer, station_layer, special_layer],
            initial_view_state=view_state,
            tooltip={"html": "<b>Station:</b> {station}", "style": {"backgroundColor": "white", "color": "black"}}
        )

        # render into placeholder
        placeholder.pydeck_chart(deck)

        # small pause to animate
        time.sleep(delay)

    # final frame: full route (ensure final line visible)
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
    if not route or not route.get("station_line1"):
        return 
    
    seg1 = route.get("station_line1", []) or []
    seg2 = route.get("station_line2", []) or []
    interchange = route.get("interchange_station", []) or []

    route_text = build_route_text(seg1, seg2, interchange)

    if not route_text:
        st.info("No route segments available.")
        return

    return st.markdown(
        f"""
        <div style="
            border:1px dashed #cfd6e4;
            border-radius:14px;
            padding:14px 16px;
            background:#fbfdff;
            box-shadow:0 2px 10px rgba(0,0,0,0.04);
            max-width:1050px;">
            <div style="font-weight:600;margin-bottom:8px;">Route</div>
            <div style="font-size:15px;">{route_text}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ---------------------------
# Render chat history (with previews for past assistant turns)
# ---------------------------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        if msg["role"] == "assistant":
            # if earlier assistant turns had data, render their previews too
            tdet = msg.get("ticket_details")
            rdet = msg.get("route_details")
            if tdet:
                render_ticket_preview(tdet)
            if rdet:
                render_route_preview(rdet)
            if msg.get("query_type"):
                st.caption(f"Query Type: {msg['query_type']}")


# ---------------------------
# Input box & new turn
# ---------------------------
if user_input := st.chat_input("Type your message here..."):
    # Show user turn
    st.session_state.messages.append({"role": "user", "content": user_input})
    st.chat_message("user").write(user_input)

    # Get AI response
    ai_reply = get_ai_reply(user_input, st.session_state.session_id)
    st.session_state.last_ai_reply = ai_reply  # for sidebar

    # Store in history (preserve details for re-render)
    st.session_state.messages.append({
        "role": "assistant",
        "content": ai_reply.get("text", ""),
        "ticket_details": ai_reply.get("ticket_details", {}),
        "route_details": ai_reply.get("route_details", {}),
        "query_type": ai_reply.get("query_type", ""),
    })

    # Render assistant bubble (current turn)
    with st.chat_message("assistant"):
        st.write(ai_reply.get("text", ""))

        # Ticket preview + download
        ticket_details = ai_reply.get("ticket_details")
        if ticket_details:
            render_ticket_preview(ticket_details)

        # Route preview
        route_details = ai_reply.get("route_details")
        if route_details:
            render_route_map(route_details, station_coords, delay=0.3)
            render_route_preview(route_details)

        if ai_reply.get("query_type"):
            st.caption(f"Query Type: {ai_reply['query_type']}")


# ---------------------------
# Sidebar: Raw JSON inspectors
# ---------------------------
with st.sidebar:
    st.header("ℹ️ Response Inspector")
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
