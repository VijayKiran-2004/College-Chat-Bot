"""
generate_metrics_pdf.py  —  College Chatbot · Metrics Reference (v2)
Generates  logs/Metrics_Reference.pdf
Usage:  python tools/generate_metrics_pdf.py
"""

import os

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT, TA_RIGHT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm, mm
from reportlab.pdfgen import canvas as pdfcanvas
from reportlab.platypus import (Frame, HRFlowable, KeepTogether, PageBreak,
                                PageTemplate, Paragraph, SimpleDocTemplate,
                                Spacer, Table, TableStyle)
from reportlab.platypus.flowables import Flowable

os.makedirs("logs", exist_ok=True)
OUTPUT = "logs/Metrics_Reference.pdf"
W, H = A4  # 595.27 x 841.89 pts

# ─────────────────────────────────────────────────────────────────────────────
# PALETTE
# ─────────────────────────────────────────────────────────────────────────────
C = {
    "navy": colors.HexColor("#0D1B2A"),
    "blue": colors.HexColor("#1565C0"),
    "blue_mid": colors.HexColor("#1976D2"),
    "blue_light": colors.HexColor("#E3F2FD"),
    "blue_pale": colors.HexColor("#F5F9FF"),
    "indigo": colors.HexColor("#283593"),
    "accent": colors.HexColor("#0288D1"),
    "teal": colors.HexColor("#00838F"),
    "green": colors.HexColor("#2E7D32"),
    "green_lt": colors.HexColor("#E8F5E9"),
    "amber": colors.HexColor("#E65100"),
    "amber_lt": colors.HexColor("#FFF3E0"),
    "red": colors.HexColor("#B71C1C"),
    "red_lt": colors.HexColor("#FFEBEE"),
    "gold": colors.HexColor("#F57F17"),
    "grey_d": colors.HexColor("#212121"),
    "grey_m": colors.HexColor("#424242"),
    "grey_l": colors.HexColor("#757575"),
    "grey_bd": colors.HexColor("#CFD8DC"),
    "grey_bg": colors.HexColor("#F5F6FA"),
    "white": colors.white,
    "divider": colors.HexColor("#B0BEC5"),
}

# ─────────────────────────────────────────────────────────────────────────────
# STYLES
# ─────────────────────────────────────────────────────────────────────────────


def ps(name, **kw):
    return ParagraphStyle(name, **kw)


ST = {
    "cover_main": ps("cm", fontSize=32, leading=40, fontName="Helvetica-Bold",
                     textColor=C["white"], alignment=TA_LEFT),
    "cover_sub": ps("cs", fontSize=14, leading=20, fontName="Helvetica",
                    textColor=colors.HexColor("#90CAF9"), alignment=TA_LEFT),
    "cover_tag": ps("ct", fontSize=10, leading=14, fontName="Helvetica",
                    textColor=colors.HexColor("#64B5F6"), alignment=TA_LEFT),

    "sec_label": ps("sl", fontSize=9, leading=12, fontName="Helvetica-Bold",
                    textColor=C["accent"], spaceBefore=0, spaceAfter=0),
    "sec_title": ps("st", fontSize=18, leading=24, fontName="Helvetica-Bold",
                    textColor=C["white"], spaceBefore=0, spaceAfter=0),

    "metric_h": ps("mh", fontSize=13, leading=18, fontName="Helvetica-Bold",
                   textColor=C["navy"]),
    "badge": ps("ba", fontSize=8, leading=10, fontName="Helvetica-Bold",
                textColor=C["white"], alignment=TA_CENTER),
    "col_lbl": ps("cl", fontSize=8, leading=10, fontName="Helvetica-Bold",
                  textColor=C["accent"], spaceBefore=6, spaceAfter=2),
    "body": ps("bo", fontSize=9.5, leading=15, fontName="Helvetica",
               textColor=C["grey_m"], alignment=TA_JUSTIFY),
    "formula_h": ps("fh", fontSize=8, leading=10, fontName="Helvetica-Bold",
                    textColor=C["white"]),
    "formula_b": ps("fb", fontSize=9.5, leading=14, fontName="Courier",
                    textColor=colors.HexColor("#E0F7FA")),
    "interp_h": ps("ih", fontSize=8, leading=10, fontName="Helvetica-Bold",
                   textColor=C["green"]),
    "interp_b": ps("ib", fontSize=9, leading=13, fontName="Helvetica",
                   textColor=C["grey_m"]),
    "note": ps("no", fontSize=8.5, leading=12, fontName="Helvetica-Oblique",
               textColor=C["grey_l"]),
    "tbl_hdr": ps("th", fontSize=9, leading=12, fontName="Helvetica-Bold",
                  textColor=C["white"]),
    "tbl_cell": ps("tc", fontSize=9, leading=13, fontName="Helvetica",
                   textColor=C["grey_m"]),
    "footer": ps("fo", fontSize=7.5, leading=10, fontName="Helvetica",
                 textColor=C["grey_l"], alignment=TA_CENTER),
    "pg_hdr": ps("ph", fontSize=8, leading=10, fontName="Helvetica-Bold",
                 textColor=C["grey_l"]),
    "overview_h": ps("oh", fontSize=11, leading=15, fontName="Helvetica-Bold",
                     textColor=C["navy"], spaceBefore=12, spaceAfter=6),
}

PAGE_W = 17 * cm   # usable width

# ─────────────────────────────────────────────────────────────────────────────
# RUNNING HEADER / FOOTER (canvas callbacks)
# ─────────────────────────────────────────────────────────────────────────────


class DocCanvas:
    def __init__(self, doc):
        self.doc = doc

    def on_page(self, canv, doc):
        pg = doc.page
        if pg == 1:
            return  # cover page — no header

        canv.saveState()
        # Top rule
        canv.setStrokeColor(C["blue_mid"])
        canv.setLineWidth(0.8)
        canv.line(2 * cm, H - 1.5 * cm, W - 2 * cm, H - 1.5 * cm)

        # Header text
        canv.setFont("Helvetica-Bold", 7.5)
        canv.setFillColor(C["grey_l"])
        canv.drawString(
            2 * cm,
            H - 1.3 * cm,
            "College Chatbot  ·  Metrics Reference Guide")
        canv.drawRightString(W - 2 * cm, H - 1.3 * cm, f"Page {pg}")

        # Bottom rule
        canv.line(2 * cm, 1.3 * cm, W - 2 * cm, 1.3 * cm)
        canv.setFont("Helvetica", 7)
        canv.drawCentredString(
            W / 2,
            0.9 * cm,
            "ACP Project  ·  Generated by tools/generate_metrics_pdf.py  ·  2026")
        canv.restoreState()


# ─────────────────────────────────────────────────────────────────────────────
# COVER PAGE (drawn directly on canvas)
# ─────────────────────────────────────────────────────────────────────────────
def draw_cover(canv, doc):
    canv.saveState()

    # Navy background
    canv.setFillColor(C["navy"])
    canv.rect(0, 0, W, H, fill=1, stroke=0)

    # Decorative diagonal stripe (top-right)
    canv.setFillColor(colors.HexColor("#112233"))
    from reportlab.graphics.shapes import Polygon
    p = canv.beginPath()
    p.moveTo(W * 0.55, H)
    p.lineTo(W, H)
    p.lineTo(W, H * 0.55)
    p.close()
    canv.drawPath(p, fill=1, stroke=0)

    # Accent bar (left edge)
    canv.setFillColor(C["accent"])
    canv.rect(0, 0, 0.5 * cm, H, fill=1, stroke=0)

    # Accent horizontal band
    canv.setFillColor(colors.HexColor("#0A2744"))
    canv.rect(0, H * 0.25, W, H * 0.08, fill=1, stroke=0)

    # College Chatbot tag
    canv.setFont("Helvetica-Bold", 10)
    canv.setFillColor(C["accent"])
    canv.drawString(2.2 * cm, H * 0.85, "COLLEGE CHATBOT  ·  ACP PROJECT")

    # Main title
    canv.setFont("Helvetica-Bold", 38)
    canv.setFillColor(C["white"])
    canv.drawString(2.2 * cm, H * 0.72, "Metrics")
    canv.drawString(2.2 * cm, H * 0.64, "Reference")

    # Accent colour word
    canv.setFillColor(C["accent"])
    canv.drawString(2.2 * cm, H * 0.56, "Guide")

    # Subtitle
    canv.setFont("Helvetica", 13)
    canv.setFillColor(colors.HexColor("#90CAF9"))
    canv.drawString(2.2 * cm, H * 0.49,
                    "A plain-English guide to every metric in the log sheets")

    # Divider line
    canv.setStrokeColor(C["accent"])
    canv.setLineWidth(1.5)
    canv.line(2.2 * cm, H * 0.46, 14 * cm, H * 0.46)

    # Metric pill labels
    pills = [
        ("TIME", C["blue_mid"]),
        ("RETRIEVAL", C["teal"]),
        ("FAITHFULNESS", C["green"]),
        ("RELEVANCE", C["amber"]),
        ("BERTSCORE", C["indigo"]),
        ("ACCURACY", C["blue"]),
    ]
    x = 2.2 * cm
    y = H * 0.40
    canv.setFont("Helvetica-Bold", 8)
    for label, col in pills:
        tw = canv.stringWidth(label, "Helvetica-Bold", 8)
        pw = tw + 16
        canv.setFillColor(col)
        canv.roundRect(x, y, pw, 14, 4, fill=1, stroke=0)
        canv.setFillColor(C["white"])
        canv.drawString(x + 8, y + 3.5, label)
        x += pw + 6

    # Version / date
    canv.setFont("Helvetica", 9)
    canv.setFillColor(colors.HexColor("#546E7A"))
    canv.drawString(2.2 * cm, H * 0.33, "Version 2.0   ·   March 2026")

    # Bottom strip
    canv.setFillColor(C["blue"])
    canv.rect(0, 0, W, 1.6 * cm, fill=1, stroke=0)
    canv.setFont("Helvetica", 8)
    canv.setFillColor(C["white"])
    canv.drawCentredString(
        W /
        2,
        0.55 *
        cm,
        "College Chatbot  ·  Evaluation Metrics Documentation  ·  ACP Project 2026")

    canv.restoreState()


# ─────────────────────────────────────────────────────────────────────────────
# SECTION DIVIDER
# ─────────────────────────────────────────────────────────────────────────────
def section_divider(number, title, subtitle=""):
    rows = []
    if subtitle:
        rows.append([Paragraph(f"SECTION {number}", ST["sec_label"]), ""])
        rows.append([Paragraph(title, ST["sec_title"]), ""])
        rows.append([Paragraph(subtitle,
                               ps("subsub",
                                  fontSize=9,
                                  leading=13,
                                  fontName="Helvetica",
                                  textColor=colors.HexColor("#90CAF9"))),
                     ""])
    else:
        rows.append([Paragraph(f"SECTION {number}", ST["sec_label"]), ""])
        rows.append([Paragraph(title, ST["sec_title"]), ""])

    t = Table(rows, colWidths=[PAGE_W - 1.5 * cm, 1.5 * cm])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), C["navy"]),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("LEFTPADDING", (0, 0), (-1, -1), 16),
        ("RIGHTPADDING", (0, 0), (-1, -1), 10),
        # Left accent bar via inner table border
        ("LINEBEFORE", (0, 0), (0, -1), 4, C["accent"]),
    ]))
    return [Spacer(1, 10), t, Spacer(1, 6)]


# ─────────────────────────────────────────────────────────────────────────────
# FORMULA BOX
# ─────────────────────────────────────────────────────────────────────────────
def formula_box(lines):
    rows = [[Paragraph("FORMULA", ST["formula_h"])]]
    for line in lines:
        rows.append([Paragraph(line if line else " ", ST["formula_b"])])
    t = Table(rows, colWidths=[PAGE_W])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), C["navy"]),
        ("BACKGROUND", (0, 0), (-1, 0), C["blue"]),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("LEFTPADDING", (0, 0), (-1, -1), 12),
        ("RIGHTPADDING", (0, 0), (-1, -1), 12),
        ("LINEABOVE", (0, 0), (-1, 0), 0, C["blue"]),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [C["navy"], colors.HexColor("#0A1929")]),
    ]))
    return t


# ─────────────────────────────────────────────────────────────────────────────
# SCORE BAND TABLE
# ─────────────────────────────────────────────────────────────────────────────
def score_bands(bands):
    """bands = list of (range_str, meaning, color_key)"""
    rows = [[
        Paragraph("SCORE", ST["col_lbl"]),
        Paragraph("INTERPRETATION", ST["col_lbl"]),
    ]]
    row_bgs = []
    for (rng, meaning, ckey) in bands:
        rows.append([
            Paragraph(f"<b>{rng}</b>",
                      ps("rb", fontSize=9, fontName="Helvetica-Bold",
                         textColor=C[ckey])),
            Paragraph(meaning, ST["tbl_cell"]),
        ])
        row_bgs.append(C["grey_bg"])

    t = Table(rows, colWidths=[3.5 * cm, PAGE_W - 3.5 * cm])
    style = [
        ("BACKGROUND", (0, 0), (-1, 0), C["grey_bg"]),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("LEFTPADDING", (0, 0), (-1, -1), 10),
        ("RIGHTPADDING", (0, 0), (-1, -1), 10),
        ("GRID", (0, 0), (-1, -1), 0.3, C["divider"]),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [C["white"], C["grey_bg"]]),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ]
    # Colour the left cell per band
    for i, (_, _, ckey) in enumerate(bands, start=1):
        style.append(("BACKGROUND", (0, i), (0, i), C[ckey + "_lt"]
                      if ckey + "_lt" in C else C["blue_light"]))
    t.setStyle(TableStyle(style))
    return t


# ─────────────────────────────────────────────────────────────────────────────
# METRIC CARD
# ─────────────────────────────────────────────────────────────────────────────
def metric_card(title, sheet_tag, accent_color,
                what, formula_lines, how, bands, note=None):
    elems = []

    # ── Header strip ─────────────────────────────────────────────────────────
    hdr_data = [[
        Paragraph(title, ST["metric_h"]),
        Paragraph(sheet_tag, ps("sh", fontSize=8, fontName="Helvetica-Bold",
                                textColor=C["white"], alignment=TA_RIGHT)),
    ]]
    hdr = Table(hdr_data, colWidths=[PAGE_W - 3 * cm, 3 * cm])
    hdr.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), C["white"]),
        ("BACKGROUND", (1, 0), (1, 0), accent_color),
        ("TOPPADDING", (0, 0), (-1, -1), 10),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
        ("LEFTPADDING", (0, 0), (0, 0), 14),
        ("RIGHTPADDING", (1, 0), (1, 0), 10),
        ("LEFTPADDING", (1, 0), (1, 0), 6),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ]))

    # ── What it measures ────────────────────────────────────────────────────
    what_data = [[
        Paragraph("WHAT IT MEASURES", ST["col_lbl"]),
    ], [
        Paragraph(what, ST["body"]),
    ]]
    what_t = Table(what_data, colWidths=[PAGE_W])
    what_t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), C["blue_pale"]),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("LEFTPADDING", (0, 0), (-1, -1), 14),
        ("RIGHTPADDING", (0, 0), (-1, -1), 14),
    ]))

    # ── How it works ────────────────────────────────────────────────────────
    how_data = [[
        Paragraph("HOW IT WORKS", ST["col_lbl"]),
    ], [
        Paragraph(how, ST["body"]),
    ]]
    how_t = Table(how_data, colWidths=[PAGE_W])
    how_t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), C["white"]),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("LEFTPADDING", (0, 0), (-1, -1), 14),
        ("RIGHTPADDING", (0, 0), (-1, -1), 14),
    ]))

    # ── Score bands ─────────────────────────────────────────────────────────
    interp_lbl = Table([[Paragraph("HOW TO READ THE RESULT", ST["col_lbl"])]],
                       colWidths=[PAGE_W])
    interp_lbl.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), C["grey_bg"]),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
        ("LEFTPADDING", (0, 0), (-1, -1), 14),
    ]))

    # ── Note ─────────────────────────────────────────────────────────────────
    note_rows = []
    if note:
        note_t = Table([[Paragraph(f"ℹ  {note}", ST["note"])]],
                       colWidths=[PAGE_W])
        note_t.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), C["amber_lt"]),
            ("TOPPADDING", (0, 0), (-1, -1), 6),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ("LEFTPADDING", (0, 0), (-1, -1), 14),
            ("RIGHTPADDING", (0, 0), (-1, -1), 14),
            ("LINEBEFORE", (0, 0), (0, -1), 3, C["gold"]),
        ]))
        note_rows = [note_t]

    # ── Outer card frame ─────────────────────────────────────────────────────
    inner = [hdr, what_t, formula_box(formula_lines),
             how_t, interp_lbl, score_bands(bands)] + note_rows

    outer = Table([[item] for item in inner], colWidths=[PAGE_W])
    outer.setStyle(TableStyle([
        ("BOX", (0, 0), (-1, -1), 0.8, C["divider"]),
        ("LINEBEFORE", (0, 0), (-1, -1), 4, accent_color),
        ("TOPPADDING", (0, 0), (-1, -1), 0),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
        ("LEFTPADDING", (0, 0), (-1, -1), 0),
        ("RIGHTPADDING", (0, 0), (-1, -1), 0),
    ]))

    elems.append(KeepTogether([Spacer(1, 10), outer]))
    return elems


# ─────────────────────────────────────────────────────────────────────────────
# BUILD DOCUMENT
# ─────────────────────────────────────────────────────────────────────────────
dc = DocCanvas(None)
doc = SimpleDocTemplate(
    OUTPUT, pagesize=A4,
    leftMargin=2 * cm, rightMargin=2 * cm,
    topMargin=2.2 * cm, bottomMargin=2 * cm,
)

story = []

# ── COVER (entirely drawn by onFirstPage canvas callback) ────────────────────
# No story items needed for page 1 — the canvas callback fills the whole page.
# forces page 1 content (the cover) to end immediately
story.append(PageBreak())

# ── PAGE-WIDE INTRO ─────────────────────────────────────────────────────
story.append(Paragraph("Overview — All Metrics", ST["overview_h"]))
story.append(Paragraph(
    "This document explains every metric recorded by the College Chatbot "
    "evaluation system. Each metric card shows <b>what it measures</b>, "
    "the <b>exact formula</b>, <b>how the calculation works</b>, and "
    "<b>how to interpret the score</b>.",
    ps("intro", fontSize=9.5, leading=15, fontName="Helvetica",
       textColor=C["grey_m"], alignment=TA_JUSTIFY, spaceAfter=10)))

# Overview glance table
ov = [
    [Paragraph(h, ST["tbl_hdr"]) for h in
     ["Metric", "Sheet", "Range", "One-line summary", "Weight in Accuracy"]],
    ["Time Taken (s)", "Production", "0 – ∞ s", "Pure server processing time", "—"],
    ["Latency (s)", "Evaluation", "0 – ∞ s", "Total client round-trip time", "—"],
    ["Server Time (s)", "Evaluation", "0 – ∞ s", "Server time echoed in API response", "—"],
    ["Retrieval Conf. %", "Both", "0 – 100 %", "Semantic match: query vs top chunk", "—"],
    ["Faithfulness %", "Evaluation", "0 – 100 %", "Are claims grounded in the context?", "30 %"],
    ["Relevance %", "Evaluation", "0 – 100 %", "Does the answer address the question?", "25 %"],
    ["Completeness %", "Evaluation", "0 – 100 %", "All parts of the question answered?", "20 %"],
    ["BERTScore F1 %", "Evaluation", "0 – 100 %", "Semantic overlap: answer vs context", "15 %"],
    ["Link Validity", "Evaluation", "X/Y Valid", "Are hyperlinks in the answer live?", "—"],
    ["Accuracy %", "Evaluation", "0 – 100 %", "Weighted final quality score", "Final"],
    ["Source", "Both", "Text label", "Which system answered: RAG / SQL / KB", "10 % (routing)"],
]
ov_t = Table(ov, colWidths=[3.8 * cm, 2.4 * cm, 2.2 * cm, 5.8 * cm, 2.8 * cm])
ov_t.setStyle(TableStyle([
    ("BACKGROUND", (0, 0), (-1, 0), C["navy"]),
    ("TEXTCOLOR", (0, 0), (-1, 0), C["white"]),
    ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
    ("FONTSIZE", (0, 0), (-1, -1), 8.5),
    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [C["white"], C["blue_pale"]]),
    ("GRID", (0, 0), (-1, -1), 0.3, C["divider"]),
    ("TOPPADDING", (0, 0), (-1, -1), 5),
    ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ("LEFTPADDING", (0, 0), (-1, -1), 8),
    ("RIGHTPADDING", (0, 0), (-1, -1), 8),
    ("ALIGN", (4, 0), (4, -1), "CENTER"),
    ("ALIGN", (1, 0), (1, -1), "CENTER"),
    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ("LINEBEFORE", (0, 0), (0, -1), 3, C["accent"]),
]))
story.append(ov_t)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — TIMING
# ══════════════════════════════════════════════════════════════════════════════
story += section_divider(1, "Timing Metrics",
                         "How long did the system take to respond?")

story += metric_card(
    title="Time Taken (s)",
    sheet_tag="PRODUCTION SHEET",
    accent_color=C["blue"],
    what=(
        "The number of seconds the server spent processing one request — "
        "from the moment the backend receives the HTTP call to the moment it "
        "finishes generating the answer. Network transfer time is NOT included."
    ),
    formula_lines=[
        "Time Taken  =  T_end  −  T_start",
        "",
        "T_start  =  instant /query endpoint receives the request (server clock)",
        "T_end    =  instant answer is fully generated (server clock)",
    ],
    how=(
        "A Python time.time() timer is started at the top of the /query handler. "
        "It stops the moment the model returns its final answer — whether that is "
        "a SQL result or LLM-generated text. The elapsed seconds are saved to the "
        "Production sheet with 4 decimal place precision."
    ),
    bands=[
        ("0 – 1 s", "Excellent — SQL / Knowledge Base instant response", "green"),
        ("1 – 4 s", "Good — RAG with fast GPU / cached model", "teal"),
        ("4 – 10 s", "Acceptable — normal LLM inference on CPU", "amber"),
        ("> 10 s", "Slow — model may be overloaded or cold-starting", "red"),
    ],
    note="Time Taken does NOT include network latency or client rendering time.",
)

story += metric_card(
    title="Latency (s)",
    sheet_tag="EVALUATION SHEET",
    accent_color=C["teal"],
    what=(
        "The total wall-clock time from when the test script sends the HTTP "
        "request to when the final response chunk lands on the client. "
        "This includes server processing time PLUS all network overhead."
    ),
    formula_lines=[
        "Latency  =  T_received  −  T_sent",
        "",
        "Both timestamps measured on the CLIENT (prompt_test.py)",
        "For streaming: T_received = arrival of the SSE 'done' event",
        "For SQL/JSON:  T_received = full HTTP body received",
    ],
    how=(
        "In prompt_test.py, time.time() is captured before requests.post() is "
        "called. For streaming responses the clock stops when the 'done' SSE "
        "event is received. The gap naturally includes all network round-trip time, "
        "which on localhost is typically < 10 ms."
    ),
    bands=[
        ("≈ Server Time", "Network overhead negligible — local / fast LAN", "green"),
        ("+ 50–200 ms", "Small network gap — normal on local machine", "teal"),
        ("+ 0.5 – 2 s", "Noticeable delay — check network or proxy", "amber"),
        ("> Server + 2 s", "Large gap — network bottleneck or buffering issue", "red"),
    ],
)

story += metric_card(
    title="Server Time (s)",
    sheet_tag="EVALUATION SHEET",
    accent_color=C["indigo"],
    what=(
        "The server-side processing time as reported by the backend and sent "
        "back inside the API response. This is identical in meaning to "
        "Time Taken (Production) but is captured inside the Evaluation sheet "
        "so both timing perspectives sit side-by-side for comparison."
    ),
    formula_lines=[
        "Server Time  =  T_end  −  T_start        (measured on server)",
        "",
        "Streaming:  sent inside the SSE 'done' event  →  { time_taken: X.XXXX }",
        "SQL/JSON:   sent in the JSON response field   →  { time_taken: X.XXXX }",
    ],
    how=(
        "The backend computes start/end time the same way as Time Taken. It "
        "serialises the result and includes time_taken in the response payload. "
        "prompt_test.py reads this field and stores it as Server Time — giving "
        "you the authoritative server number without any client-side noise."
    ),
    bands=[
        ("≈ Latency", "Network is fast — values will be nearly identical", "green"),
        ("Much < Latency", "Network delay is significant; focus on Server Time", "teal"),
        ("Much > Latency", "This should not happen — check for clock skew", "red"),
        ("N/A", "Response did not include time_taken field", "amber"),
    ],
)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — RETRIEVAL
# ══════════════════════════════════════════════════════════════════════════════
story.append(PageBreak())
story += section_divider(2, "Retrieval Quality Metrics",
                         "Did the system find the right information?")

story += metric_card(
    title="Retrieval Confidence (%)",
    sheet_tag="BOTH SHEETS",
    accent_color=C["accent"],
    what=(
        "How confident the RAG retriever was when selecting the document "
        "chunks to answer a query. A high score means the retrieved content "
        "is semantically very close to the question. A low score means the "
        "retriever was uncertain and may have returned loosely related material."
    ),
    formula_lines=[
        "               A · B",
        "cos(θ)  =  ─────────────────",
        "             |A|  ×  |B|",
        "",
        "A · B  =  Σ (aᵢ × bᵢ)     ← dot product (multiply pairs, then sum)",
        "|A|    =  √(a₁² + a₂² + ... + aₙ²)  ← vector length",
        "",
        "θ = 0°  → cos = 1.0  → identical meaning  (score = 100%)",
        "θ = 90° → cos = 0.0  → completely unrelated (score =   0%)",
        "",
        "Confidence %  =  cos(θ)  ×  100",
    ],
    how=(
        "Both the user's query and every stored document chunk are converted to "
        "dense 384-dimension vectors using the all-MiniLM-L6-v2 sentence-transformer "
        "model. Cosine similarity measures the angle between the query vector and "
        "each chunk vector — zero angle means identical meaning, 90° means unrelated. "
        "The similarity of the single best-matching chunk is taken, multiplied by 100, "
        "and stored as Retrieval Confidence."
    ),
    bands=[
        ("80 – 100 %", "Excellent — retriever found exactly the right content", "green"),
        ("60 – 79 %", "Good — likely relevant, minor gaps possible", "teal"),
        ("40 – 59 %", "Moderate — answer may contain hallucinated details", "amber"),
        ("< 40 %", "Poor — retriever is guessing; treat answer with caution", "red"),
    ],
    note="SQL and Knowledge-Base answers show N/A — no vector retrieval is performed for structured data queries.",
)

story += metric_card(
    title="Source",
    sheet_tag="BOTH SHEETS",
    accent_color=C["blue_mid"],
    what=(
        "A text label identifying which internal system produced the answer. "
        "This tells you the exact data pipeline the chatbot used."
    ),
    formula_lines=[
        "RAG    →  Vector retrieval from PDF/HTML knowledge base + LLM generation",
        "SQL    →  Structured query on the student/placement database",
        "KB     →  Hard-coded Knowledge Base (known college facts)",
        "Hybrid →  Combination of SQL result + RAG context",
        "",
        "No formula — routing is rule-based + intent classification",
    ],
    how=(
        "The QueryRouter analyses the intent of every incoming question using "
        "keyword and embedding-based classification. Count / list questions go to "
        "SQL. Factual, procedural, and policy questions go to RAG or KB. "
        "The routing decision is attached to every log row."
    ),
    bands=[
        ("RAG", "LLM-generated answer from retrieved document chunks", "blue"),
        ("SQL", "Direct database lookup — high precision for numbers", "teal"),
        ("KB", "Hardcoded facts — fastest and most reliable", "green"),
        ("Unknown", "Router could not classify the query — review logs", "red"),
    ],
)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — LLM JUDGE
# ══════════════════════════════════════════════════════════════════════════════
story.append(PageBreak())
story += section_divider(3, "LLM-as-Judge Metrics",
                         "Scored by Gemma 1B using the question, context, and answer")

# Judge intro box
intro_t = Table([[Paragraph(
    "The three metrics below are evaluated by a small local language model (Gemma 1B via Ollama). "
    "The judge receives the original question, the retrieved context (up to 600 chars), and "
    "the chatbot's answer (up to 600 chars). It returns a JSON object with three 0.0–1.0 scores. "
    "These are multiplied by 100 and saved as percentages.",
    ps("ji", fontSize=9, leading=14, fontName="Helvetica-Oblique",
       textColor=C["navy"], alignment=TA_JUSTIFY)
)]], colWidths=[PAGE_W])
intro_t.setStyle(TableStyle([
    ("BACKGROUND", (0, 0), (-1, -1), C["blue_light"]),
    ("TOPPADDING", (0, 0), (-1, -1), 10),
    ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
    ("LEFTPADDING", (0, 0), (-1, -1), 14),
    ("RIGHTPADDING", (0, 0), (-1, -1), 14),
    ("LINEBEFORE", (0, 0), (0, -1), 4, C["blue"]),
]))
story.append(Spacer(1, 8))
story.append(intro_t)

story += metric_card(
    title="Faithfulness % (LLM)",
    sheet_tag="EVALUATION SHEET",
    accent_color=C["green"],
    what=(
        "Whether every factual claim in the chatbot's answer is directly "
        "supported by the retrieved context. An answer that invents facts not "
        "present in the source material will score low — even if those facts "
        "happen to be correct in the real world."
    ),
    formula_lines=[
        "Judge prompt:  'Is every claim in the answer supported by the context?'",
        "Judge returns: { \"faithfulness\": <0.0 – 1.0> }",
        "",
        "  Faithfulness %  =  judge_score  ×  100",
        "",
        "No explicit math — Gemma 1B reads context + answer and outputs",
        "a strict 0.0–1.0 opinion on whether every claim is grounded.",
        "",
        "Weight in Accuracy: 30 %  (highest — hallucination is biggest risk)",
    ],
    how=(
        "The Gemma judge reads the retrieved context and the bot answer side by "
        "side. For each factual statement in the answer it checks: can this be "
        "traced to the context? Filler phrases like 'please contact the office' "
        "are ignored. Only verifiable claims are assessed. A score of 1.0 means "
        "every single claim has clear contextual support."
    ),
    bands=[
        ("90 – 100 %", "Every statement grounded in source — highly reliable", "green"),
        ("70 – 89 %", "Mostly faithful, minor unsupported details present", "teal"),
        ("50 – 69 %", "Some hallucination — review the answer carefully", "amber"),
        ("< 50 %", "Significant hallucination — retrieval may be wrong", "red"),
    ],
)

story += metric_card(
    title="Relevance % (LLM)",
    sheet_tag="EVALUATION SHEET",
    accent_color=C["blue_mid"],
    what=(
        "How directly the chatbot's answer addresses the actual question asked. "
        "A high-relevance answer stays on topic. A low-relevance answer may be "
        "truthful but discusses something other than what the user asked."
    ),
    formula_lines=[
        "Judge prompt:  'Does the answer directly address the question?'",
        "Judge returns: { \"relevance\": <0.0 – 1.0> }",
        "",
        "  Relevance %  =  judge_score  ×  100",
        "",
        "No explicit math — Gemma 1B reads question + answer and rates",
        "how directly the response addresses what was asked.",
        "",
        "Weight in Accuracy: 25 %",
    ],
    how=(
        "The judge is shown the question and answer (no context). It evaluates: "
        "did the response stay on topic and specifically answer what was asked? "
        "For example — if a student asks 'What is the library timing?' and the "
        "bot responds about fee structure, that is a near-zero relevance answer "
        "regardless of how accurate the fee information is."
    ),
    bands=[
        ("90 – 100 %", "Laser-focused — exactly what the user asked for", "green"),
        ("70 – 89 %", "Mostly on-topic with minor tangents", "teal"),
        ("50 – 69 %", "Partly relevant — answer drifts into unrelated info", "amber"),
        ("< 50 %", "Bot answered the wrong question entirely", "red"),
    ],
)

story += metric_card(
    title="Completeness % (LLM)",
    sheet_tag="EVALUATION SHEET",
    accent_color=C["teal"],
    what=(
        "Whether the answer covers ALL aspects of the question, not just part "
        "of it. A complete answer addresses every sub-part of a multi-faceted "
        "question. Partial answers score lower even if the covered portion is "
        "perfectly accurate and relevant."
    ),
    formula_lines=[
        "Judge prompt:  'Does the answer cover all aspects of the question?'",
        "Judge returns: { \"completeness\": <0.0 – 1.0> }",
        "",
        "  Completeness %  =  judge_score  ×  100",
        "",
        "No explicit math — Gemma 1B maps the scope of the question against",
        "the scope of the answer and rates coverage 0.0 – 1.0.",
        "",
        "Weight in Accuracy: 20 %",
    ],
    how=(
        "The judge maps the scope of the question against the scope of the answer. "
        "Example: 'Tell me about placement statistics' expects total placed, company "
        "names, package ranges, and department breakdown. An answer mentioning only "
        "the total count scores around 0.3. All four aspects covered scores near 1.0."
    ),
    bands=[
        ("90 – 100 %", "Comprehensive — all parts of the question answered", "green"),
        ("70 – 89 %", "Mostly complete — one or two minor aspects missed", "teal"),
        ("50 – 69 %", "Partial answer — significant parts unanswered", "amber"),
        ("< 50 %", "Only a small fraction of the question was addressed", "red"),
    ],
)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — STATISTICAL
# ══════════════════════════════════════════════════════════════════════════════
story.append(PageBreak())
story += section_divider(4, "Statistical & Validity Metrics",
                         "Objective, formula-based checks that complement the LLM judge")

story += metric_card(
    title="BERTScore F1 %",
    sheet_tag="EVALUATION SHEET",
    accent_color=C["indigo"],
    what=(
        "A semantic similarity score measuring how closely the chatbot's answer "
        "mirrors the retrieved context at the sentence level. Unlike simple word "
        "matching, it understands meaning — 'students passed' and 'learners "
        "succeeded' are treated as similar even though no words overlap."
    ),
    formula_lines=[
        "Step 1 — split answer and context into individual sentences",
        "",
        "               1   n",
        "Precision  =  ─── Σ  max  cos(answer_sent_i , context_sent_j)",
        "               n  i=1   j",
        "",
        "               1   m",
        "Recall     =  ─── Σ  max  cos(context_sent_j , answer_sent_i)",
        "               m  j=1   i",
        "",
        "               2 × Precision × Recall",
        "F1         =  ────────────────────────",
        "               Precision  +  Recall",
        "",
        "BERTScore F1 %  =  F1  ×  100",
        "Weight in Accuracy: 15 %",
    ],
    how=(
        "Both the answer and the retrieved context are split into individual sentences. "
        "Each sentence is encoded into a 384-dimension vector by all-MiniLM-L6-v2. "
        "A similarity matrix is built between all answer sentences and all context "
        "sentences. Precision measures how well the answer covers the context. "
        "Recall measures how well the context is reflected in the answer. "
        "F1 balances both into a single number."
    ),
    bands=[
        ("85 – 100 %", "Very strong overlap — answer closely mirrors the docs", "green"),
        ("70 – 84 %", "Good match — some paraphrasing or minor omissions", "teal"),
        ("50 – 69 %", "Moderate — answer uses very different language to docs", "amber"),
        ("< 50 %", "Answer is semantically distant — possible hallucination", "red"),
    ],
    note="For SQL/KB answers where context is empty, this defaults to 100 % — evaluated by the LLM judge instead.",
)

story += metric_card(
    title="Link Validity",
    sheet_tag="EVALUATION SHEET",
    accent_color=C["amber"],
    what=(
        "Whether the hyperlinks included in the chatbot's response actually "
        "work. Broken links frustrate users and indicate that knowledge-base "
        "content needs updating. This metric counts how many links are reachable."
    ),
    formula_lines=[
        "Links found  =  regex: r'\\[.*?\\]\\((https?://.*?)\\)'",
        "Valid links  =  links returning HTTP status code < 400 within 5 s",
        "Score        =  valid_count  ÷  total_count",
        "Display      =  'X/Y Valid'  e.g. '3/4 Valid'",
        "Weight in Accuracy: 0 %  (informational only — links are rare)",
    ],
    how=(
        "After the bot answer is received, the script scans for markdown-format "
        "URLs. For each one, an HTTP HEAD request is sent — this is lightweight "
        "and only checks that the page exists, without downloading its content. "
        "A response of 200–399 marks the link as Valid. Timeouts, 404s, or 5xx "
        "errors mark it as broken. Answers with no links show 'N/A'."
    ),
    bands=[
        ("N/A", "No hyperlinks in the answer — normal for SQL/KB answers", "teal"),
        ("All Valid", "All links live — references are up to date", "green"),
        ("Partial", "Some links broken — update those pages in the knowledge base", "amber"),
        ("0/N Valid", "All links broken — knowledge base may have stale URLs", "red"),
    ],
)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — FINAL ACCURACY
# ══════════════════════════════════════════════════════════════════════════════
story.append(PageBreak())
story += section_divider(5, "Final Accuracy Score",
                         "One balanced number summarising overall response quality")

story += metric_card(
    title="Accuracy %",
    sheet_tag="EVALUATION SHEET",
    accent_color=C["blue"],
    what=(
        "A single weighted score summarising the overall quality of one chatbot "
        "response. It combines four independent quality signals — each measuring "
        "a different dimension — so no single factor can mask problems in another. "
        "Easy to compare across runs, prompts, and model versions."
    ),
    formula_lines=[
        "Accuracy  =  Σ  wᵢ · xᵢ   (weighted mean, all xᵢ in [0.0, 1.0])",
        "",
        "  w₁ · Faithfulness    =  0.30  ×  x₁",
        "  w₂ · Relevance       =  0.25  ×  x₂",
        "  w₃ · Completeness    =  0.20  ×  x₃",
        "  w₄ · BERTScore F1    =  0.15  ×  x₄",
        "  w₅ · Source Score    =  0.10  ×  x₅",
        "                         ─────────────────",
        "  Σ wᵢ                 =  1.00  (weights always sum to 1)",
        "",
        "Accuracy %  =  Accuracy  ×  100",
        "",
        "Source Score: 1.0 = correct pipeline | 0.75 = neutral | 0.0 = unknown",
    ],
    how=(
        "Faithfulness carries the highest weight (30 %) because a hallucinated "
        "answer is worse than an incomplete one. Relevance (25 %) rewards "
        "staying on topic. Completeness (20 %) rewards coverage. BERTScore (15 %) "
        "adds an objective statistical counterweight to the LLM judge. "
        "Source Appropriateness (10 %) gives a small bonus when the right pipeline "
        "handled the query — e.g., SQL for 'list all students', RAG for policies."
    ),
    bands=[
        ("90 – 100 %", "Excellent — accurate, complete, grounded, on-topic", "green"),
        ("75 – 89 %", "Good quality — suitable for real-user deployment", "teal"),
        ("60 – 74 %", "Acceptable — some improvements needed", "amber"),
        ("40 – 59 %", "Poor — investigate retrieval failures or prompt design", "amber"),
        ("< 40 %", "Problematic — likely hallucinating or mis-routing", "red"),
    ],
    note="Weights can be adjusted in prompt_test.py to reflect your team's priorities.",
)

# Weight table
story.append(Spacer(1, 10))
story.append(Paragraph("<b>Weight Breakdown</b>",
                       ps("wb_h", fontSize=10, fontName="Helvetica-Bold",
                          textColor=C["navy"], spaceBefore=4, spaceAfter=6)))
w_rows = [
    [Paragraph(h, ST["tbl_hdr"]) for h in ["Metric", "Weight", "Reason"]],
    ["Faithfulness %", "30 %", "Hallucination is the biggest risk in RAG systems"],
    ["Relevance %", "25 %", "Off-topic answers are useless even if accurate"],
    ["Completeness %", "20 %", "Partial answers leave students without key info"],
    ["BERTScore F1 %", "15 %", "Objective statistical check alongside LLM judge"],
    ["Source Score", "10 %", "Minor routing bonus — correct pipeline improves trust"],
    [Paragraph("<b>TOTAL</b>", ps("wt", fontSize=9, fontName="Helvetica-Bold",
               textColor=C["navy"])),
     Paragraph("<b>100 %</b>", ps("wp", fontSize=9, fontName="Helvetica-Bold",
               textColor=C["navy"])), ""],
]
w_t = Table(w_rows, colWidths=[3.8 * cm, 2 * cm, 11.2 * cm])
w_t.setStyle(TableStyle([
    ("BACKGROUND", (0, 0), (-1, 0), C["navy"]),
    ("TEXTCOLOR", (0, 0), (-1, 0), C["white"]),
    ("BACKGROUND", (0, -1), (-1, -1), C["blue_light"]),
    ("FONTSIZE", (0, 0), (-1, -1), 9),
    ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
    ("ROWBACKGROUNDS", (0, 1), (-1, -2), [C["white"], C["grey_bg"]]),
    ("GRID", (0, 0), (-1, -1), 0.3, C["divider"]),
    ("TOPPADDING", (0, 0), (-1, -1), 6),
    ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ("LEFTPADDING", (0, 0), (-1, -1), 10),
    ("ALIGN", (1, 0), (1, -1), "CENTER"),
    ("LINEBEFORE", (0, 0), (0, -1), 3, C["blue"]),
]))
story.append(w_t)


# ─────────────────────────────────────────────────────────────────────────────
# BUILD with running header/footer + cover canvas
# ─────────────────────────────────────────────────────────────────────────────
def on_page(canv, doc):
    if doc.page == 1:
        draw_cover(canv, doc)
    else:
        dc.on_page(canv, doc)


doc.build(story, onFirstPage=on_page, onLaterPages=on_page)
print(f"\n✓  PDF generated → {OUTPUT}")
print(f"   Sections: Timing | Retrieval | LLM Judge | Statistical | Final Accuracy")
