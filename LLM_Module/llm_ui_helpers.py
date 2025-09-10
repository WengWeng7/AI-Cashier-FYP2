import io
import qrcode
import json
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib import colors
from reportlab.lib.units import mm

def generate_ticket_pdf(ticket: dict) -> bytes:
    """
    Creates a modern RapidKL-style ticket inspired by train ticket designs.
    Bold header/footer, clean middle section, QR on left, details on right.
    """
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=letter)
    width, height = letter

    # Ticket size
    card_w = 160 * mm
    card_h = 70 * mm
    margin_x = (width - card_w) / 2
    margin_y = height - (card_h + 60)

    # Background
    c.setFillColor(colors.whitesmoke)
    c.roundRect(margin_x, margin_y, card_w, card_h, 4 * mm, fill=1, stroke=0)

    # Top header (red band)
    c.setFillColorRGB(0.85, 0.0, 0.0)  # RapidKL red
    c.rect(margin_x, margin_y + card_h - 15 * mm, card_w, 15 * mm, fill=1, stroke=0)
    c.setFont("Helvetica-Bold", 14)
    c.setFillColor(colors.white)
    c.drawString(margin_x + 8 * mm, margin_y + card_h - 8 * mm, "RapidKL Ticket")

    # Footer (blue band) → reduced height for more breathing space
    footer_h = 6 * mm
    c.setFillColorRGB(0.0, 0.3, 0.7)  # RapidKL blue
    c.rect(margin_x, margin_y, card_w, footer_h, fill=1, stroke=0)
    c.setFont("Helvetica-Oblique", 9)
    c.setFillColor(colors.white)
    c.drawRightString(margin_x + card_w - 6 * mm, margin_y + (footer_h/2) - 3, "Enjoy your journey with RapidKL!")

    # QR Code (left side)
    qr_data = json.dumps(ticket, ensure_ascii=False)
    qr_img = qrcode.make(qr_data)
    qr_size = 38 * mm
    qr_x = margin_x + 8 * mm
    qr_y = margin_y + (card_h - qr_size) / 2 - 2  # nudged slightly to center better
    c.drawInlineImage(qr_img, qr_x, qr_y, qr_size, qr_size)

    # Ticket info (right side)
    info_x = qr_x + qr_size + 15
    y = margin_y + card_h - 20 * mm
    line_gap = 7 * mm

    def field(label, value, bold=False):
        nonlocal y
        display_value = value if value else "-"  # fallback dash
        if bold:
            c.setFont("Helvetica-Bold", 12)
        else:
            c.setFont("Helvetica", 10)
        c.setFillColor(colors.black)
        c.drawString(info_x, y, f"{label}: {display_value}")
        y -= line_gap

    # Fields layout
    field("Ticket ID", ticket.get("ticket_id", ""), bold=True)
    field("From", ticket.get("from_station", ""), bold=True)
    field("To", ticket.get("to_station", ""), bold=True)
    field("Fare", ticket.get("fare", ""))
    field("Session ID", ticket.get("session_id", ""))
    field("Interchange", ticket.get("interchange", ""))
    field("Date/Time", ticket.get("datetime", ""))

    c.showPage()
    c.save()
    buf.seek(0)
    return buf.read()