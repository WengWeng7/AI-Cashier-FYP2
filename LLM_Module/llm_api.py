from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional, Dict
from ollama_llm import run_llm, kiosk_station
#from hf_llm import run_llm, kiosk_station

# QR ticket dependencies
import io
import uuid
from fastapi.responses import StreamingResponse
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import mm
from reportlab.lib import colors

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
    station_line1: List[str] = []
    station_line2: List[str] = []
    interchange_station: List[str] = []

class LLMResponse(BaseModel):
    text: str
    ticket_details: Optional[TicketDetails] = None
    route_details: Optional[RouteDetails] = None
    query_type: Optional[str] = None

class LLMRequest(BaseModel):
    user_message: str
    session_id: str

# ==== Routes ====

@app.get("/")
def root():
    return {"message": f"Welcome to AI Ticketing Kiosk API. You are at {kiosk_station}."}

@app.post("/chat", response_model=LLMResponse)
def chat(req: LLMRequest):
    result_json = run_llm(req.user_message, req.session_id)

    # Ensure all fields exist even if run_llm returns None
    response_data = {
        "text": result_json.get("text", ""),
        "ticket_details": result_json.get("ticket_details") or TicketDetails(),
        "route_details": result_json.get("route_details") or RouteDetails(),
        "query_type": result_json.get("query_type", "")
    }

    return response_data

# ==== PDF download endpoint ====
@app.get("/download/{token}")
def download_ticket(token: str):
    pdf_bytes = pdf_storage.get(token)
    if not pdf_bytes:
        return {"error": "Invalid or expired token."}
    return StreamingResponse(io.BytesIO(pdf_bytes),
                             media_type="application/pdf",
                             headers={"Content-Disposition": f"attachment; filename=ticket_{token}.pdf"})