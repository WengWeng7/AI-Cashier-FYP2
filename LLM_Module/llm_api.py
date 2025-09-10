from fastapi import FastAPI, WebSocket
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import asyncio
import io
from pydantic import BaseModel
from typing import List, Optional, Dict
from ollama_llm import run_llm, kiosk_station, fare_system, FareSystem
import ollama_llm
from llm_ui_helpers import generate_ticket_pdf
from station import lines_data
from station_coords import station_coords

# Temporary in-memory store for PDFs
pdf_storage: Dict[str, bytes] = {}

app = FastAPI(title="AI Ticketing Kiosk API")

# ==== Request & Response Models ====

class TicketDetails(BaseModel):
    session_id: str = ""
    ticket_id: str = ""
    from_station: str = ""
    to_station: str = ""
    fare: str = ""
    interchange: str = ""
    datetime: str = ""

class RouteDetails(BaseModel):
    station_lines: List[List[str]] = []
    interchanges: List[str] = []

class LLMResponse(BaseModel):
    text: str
    ticket_details: Optional[TicketDetails] = None
    route_details: Optional[RouteDetails] = None
    query_type: Optional[str] = None

class LLMRequest(BaseModel):
    user_message: str
    session_id: str
    
class StationUpdateRequest(BaseModel):
    station_name: str

# === Enable CORS ===
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==== Routes ====

@app.get("/")
def root():
    import ollama_llm
    return {"message": f"Welcome to AI Ticketing Kiosk API. You are at {ollama_llm.kiosk_station}."}

@app.post("/chat", response_model=LLMResponse)
def chat(req: LLMRequest):
    print(f"LLM API received user_message: {repr(req.user_message)} session_id: {req.session_id}")
    result_json = run_llm(req.user_message, req.session_id)

    # Ensure all fields exist even if run_llm returns None
    response_data = {
        "text": result_json.get("text", ""),
        "ticket_details": result_json.get("ticket_details") or TicketDetails(),
        "route_details": result_json.get("route_details") or RouteDetails(),
        "query_type": result_json.get("query_type", "")
    }

    return response_data

@app.get("/current_station")
def get_current_station():
    import ollama_llm
    return {"current_station": ollama_llm.kiosk_station}

@app.websocket("/ws/current_station")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    last_station = None
    while True:
        # Replace with your own logic to detect station changes
        current_station = ollama_llm.kiosk_station
        if current_station != last_station:
            await websocket.send_json({"current_station": current_station})
            last_station = current_station
        await asyncio.sleep(1)  # Check every second

@app.post("/set_station")
def set_station(req: StationUpdateRequest):
    global kiosk_station, fare_system

    # Update kiosk_station inside ollama_llm module
    ollama_llm.kiosk_station = req.station_name
    ollama_llm.fare_system = FareSystem("Fare.csv", from_station=req.station_name)

    return {"message": f"Kiosk station updated to {req.station_name}"}

@app.get("/lines_data")
def get_lines_data():
    return {"lines_data": lines_data}

@app.post("/generate_ticket")
def generate_ticket(ticket: TicketDetails):
    # Generate PDF from provided ticket details
    pdf_bytes = generate_ticket_pdf(ticket.dict())

    # Stream the PDF back directly
    return StreamingResponse(
        io.BytesIO(pdf_bytes),
        media_type="application/pdf",
        headers={
            "Content-Disposition": f"inline; filename={ticket.ticket_id or 'ticket'}.pdf"
        }
    )
    
@app.post("/route_map")
def get_route_map(route_details: RouteDetails):
    """
    Build structured JSON for route map instead of rendering in Python.
    React frontend will use this with deck.gl.
    """
    station_lines = route_details.get("station_lines", []) or []
    interchanges = route_details.get("interchanges", []) or []
    coordinates = station_coords  # from station_coords.py

    # Build full ordered list of stations
    full_names = []
    for idx, seg in enumerate(station_lines):
        if idx > 0 and idx-1 < len(interchanges):
            inter = interchanges[idx-1]
            if full_names and full_names[-1] != inter:
                full_names.append(inter)
        full_names.extend(seg)

    # Collect coords
    coords = []
    stations = []
    missing = []
    
    for i, name in enumerate(full_names):
        pt = coordinates.get(name)
        if pt:
            lat, lon = pt
            coords.append([lon, lat])

            # Label station type
            if i == 0:
                stype = "start"
            elif i == len(full_names) - 1:
                stype = "end"
            elif name in interchanges:
                stype = "interchange"
            else:
                stype = "normal"

            stations.append({
                "name": name,
                "lat": lat,
                "lon": lon,
                "type": stype
            })
        else:
            missing.append(name)

    if len(coords) < 2:
        return {"error": "Not enough station coordinates to draw a route."}

    return {
        "stations": stations,
        "path": coords,
        "missing_stations": missing
    }