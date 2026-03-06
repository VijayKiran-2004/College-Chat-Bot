"""
generate_metrics_pdf.py  —  College Chatbot · Metrics Reference (v3: Objective Only)
Generates  logs/Metrics_Reference.pdf
Usage:  python tools/generate_metrics_pdf.py
"""
import os
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import cm, mm
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_RIGHT, TA_JUSTIFY
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    HRFlowable, KeepTogether, PageBreak, Frame, PageTemplate
)
from reportlab.platypus.flowables import Flowable
from reportlab.pdfgen import canvas as pdfcanvas

os.makedirs("logs", exist_ok=True)
OUTPUT = "logs/Metrics_Reference.pdf"
W, H = A4

# ─────────────────────────────────────────────────────────────────────────────
# PALETTE
# ─────────────────────────────────────────────────────────────────────────────
C = {
    "navy":      colors.HexColor("#0D1B2A"),
    "blue":      colors.HexColor("#1565C0"),
    "blue_mid":  colors.HexColor("#1976D2"),
    "blue_light":colors.HexColor("#E3F2FD"),
    "blue_pale": colors.HexColor("#F5F9FF"),
    "indigo":    colors.HexColor("#283593"),
    "accent":    colors.HexColor("#0288D1"),
    "teal":      colors.HexColor("#00838F"),
    "green":     colors.HexColor("#2E7D32"),
    "green_lt":  colors.HexColor("#E8F5E9"),
    "amber":     colors.HexColor("#E65100"),
    "amber_lt":  colors.HexColor("#FFF3E0"),
    "red":       colors.HexColor("#B71C1C"),
    "red_lt":    colors.HexColor("#FFEBEE"),
    "gold":      colors.HexColor("#F57F17"),
    "grey_d":    colors.HexColor("#212121"),
    "grey_m":    colors.HexColor("#424242"),
    "grey_l":    colors.HexColor("#757575"),
    "grey_bd":   colors.HexColor("#CFD8DC"),
    "grey_bg":   colors.HexColor("#F5F6FA"),
    "white":     colors.white,
    "divider":   colors.HexColor("#B0BEC5"),
}

# ─────────────────────────────────────────────────────────────────────────────
# STYLES
# ─────────────────────────────────────────────────────────────────────────────
def ps(name, **kw): return ParagraphStyle(name, **kw)
ST = {
    "cover_main": ps("cm", fontSize=32, leading=40, fontName="Helvetica-Bold", textColor=C["white"]),
    "metric_h":   ps("mh", fontSize=13, leading=18, fontName="Helvetica-Bold", textColor=C["navy"]),
    "col_lbl":    ps("cl", fontSize=8, leading=10, fontName="Helvetica-Bold", textColor=C["accent"], spaceBefore=6, spaceAfter=2),
    "body":       ps("bo", fontSize=9.5, leading=15, fontName="Helvetica", textColor=C["grey_m"], alignment=TA_JUSTIFY),
    "formula_h":  ps("fh", fontSize=8, leading=10, fontName="Helvetica-Bold", textColor=C["white"]),
    "formula_b":  ps("fb", fontSize=9.5, leading=14, fontName="Courier", textColor=colors.HexColor("#E0F7FA")),
    "note":       ps("no", fontSize=8.5, leading=12, fontName="Helvetica-Oblique", textColor=C["grey_l"]),
    "tbl_hdr":    ps("th", fontSize=9, leading=12, fontName="Helvetica-Bold", textColor=C["white"]),
    "tbl_cell":   ps("tc", fontSize=9, leading=13, fontName="Helvetica", textColor=C["grey_m"]),
    "sec_label":  ps("sl", fontSize=9, leading=12, fontName="Helvetica-Bold", textColor=C["accent"]),
    "sec_title":  ps("st", fontSize=18, leading=24, fontName="Helvetica-Bold", textColor=C["white"]),
    "overview_h": ps("oh", fontSize=11, leading=15, fontName="Helvetica-Bold", textColor=C["navy"], spaceBefore=12, spaceAfter=6),
}
PAGE_W = 17 * cm

# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def section_divider(number, title, subtitle=""):
    rows = [[Paragraph(f"SECTION {number}", ST["sec_label"]), ""], [Paragraph(title, ST["sec_title"]), ""]]
    if subtitle: rows.append([Paragraph(subtitle, ps("ss", fontSize=9, leading=13, fontName="Helvetica", textColor=colors.HexColor("#90CAF9"))), ""])
    t = Table(rows, colWidths=[PAGE_W - 1.5*cm, 1.5*cm])
    t.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,-1),C["navy"]),("TOPPADDING",(0,0),(-1,-1),6),("BOTTOMPADDING",(0,0),(-1,-1),6),("LEFTPADDING",(0,0),(-1,-1),16),("LINEBEFORE",(0,0),(0,-1),4,C["accent"])]))
    return [Spacer(1, 10), t, Spacer(1, 6)]

def formula_box(lines):
    rows = [[Paragraph("FORMULA", ST["formula_h"])]]
    for l in lines: rows.append([Paragraph(l if l else " ", ST["formula_b"])])
    t = Table(rows, colWidths=[PAGE_W])
    t.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,-1),C["navy"]),("BACKGROUND",(0,0),(-1,0),C["blue"]),("TOPPADDING",(0,0),(-1,-1),5),("LEFTPADDING",(0,0),(-1,-1),12)]))
    return t

def score_bands(bands):
    rows = [[Paragraph("SCORE", ST["col_lbl"]), Paragraph("INTERPRETATION", ST["col_lbl"])]]
    for (r,m,ck) in bands: rows.append([Paragraph(f"<b>{r}</b>", ps("rb", fontSize=9, textColor=C[ck])), Paragraph(m, ST["tbl_cell"])])
    t = Table(rows, colWidths=[3.5*cm, PAGE_W - 3.5*cm])
    t.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,0),C["grey_bg"]), ("GRID",(0,0),(-1,-1),0.3,C["divider"]), ("VALIGN",(0,0),(-1,-1),"MIDDLE")]))
    return t

def metric_card(title, sheet_tag, accent_color, what, formula_lines, how, bands, note=None):
    hdr = Table([[Paragraph(title, ST["metric_h"]), Paragraph(sheet_tag, ps("sh", fontSize=8, textColor=colors.white, alignment=TA_RIGHT))]], colWidths=[PAGE_W-3*cm, 3*cm])
    hdr.setStyle(TableStyle([("BACKGROUND",(0,0),(0,-0),colors.white),("BACKGROUND",(1,0),(1,0),accent_color),("TOPPADDING",(0,0),(-1,-1),10), ("LEFTPADDING",(0,0),(0,0),14)]))
    what_t = Table([[Paragraph("WHAT IT MEASURES", ST["col_lbl"])],[Paragraph(what, ST["body"])]], colWidths=[PAGE_W])
    what_t.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,-1),C["blue_pale"]), ("LEFTPADDING",(0,0),(-1,-1),14)]))
    how_t = Table([[Paragraph("HOW IT WORKS", ST["col_lbl"])],[Paragraph(how, ST["body"])]], colWidths=[PAGE_W])
    how_t.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,-1),colors.white), ("LEFTPADDING",(0,0),(-1,-1),14)]))
    note_t = [Table([[Paragraph(f"ℹ {note}", ST["note"])]], colWidths=[PAGE_W])] if note else []
    if note_t: note_t[0].setStyle(TableStyle([("BACKGROUND",(0,0),(-1,-1),C["amber_lt"]),("LINEBEFORE",(0,0),(0,-1),3,C["gold"]),("LEFTPADDING",(0,0),(-1,-1),14)]))
    inner = [hdr, what_t, formula_box(formula_lines), how_t, score_bands(bands)] + note_t
    outer = Table([[item] for item in inner], colWidths=[PAGE_W])
    outer.setStyle(TableStyle([("BOX",(0,0),(-1,-1),0.8,C["divider"]),("LINEBEFORE",(0,0),(-1,-1),4,accent_color),("TOPPADDING",(0,0),(-1,-1),0)]))
    return [KeepTogether([Spacer(1, 10), outer])]

# ─────────────────────────────────────────────────────────────────────────────
# COVER & HEADER
# ─────────────────────────────────────────────────────────────────────────────
def draw_cover(canv, doc):
    canv.saveState()
    canv.setFillColor(C["navy"]); canv.rect(0,0,W,H,fill=1)
    canv.setFillColor(C["accent"]); canv.rect(0,0,0.5*cm,H,fill=1)
    canv.setFont("Helvetica-Bold", 38); canv.setFillColor(C["white"]); canv.drawString(2.2*cm, H*0.72, "Objective")
    canv.drawString(2.2*cm, H*0.64, "Metrics")
    canv.setFillColor(C["accent"]); canv.drawString(2.2*cm, H*0.56, "Reference")
    canv.setFont("Helvetica", 13); canv.setFillColor(colors.HexColor("#90CAF9")); canv.drawString(2.2*cm, H*0.49, "Simplified 'No-Judge' evaluation for the College Chatbot")
    canv.restoreState()

def on_page(canv, doc):
    if doc.page == 1: draw_cover(canv, doc)
    else:
        canv.saveState(); canv.setStrokeColor(C["blue_mid"]); canv.line(2*cm, H-1.5*cm, W-2*cm, H-1.5*cm); canv.setFont("Helvetica-Bold", 7.5); canv.setFillColor(C["grey_l"])
        canv.drawString(2*cm, H-1.3*cm, "College Chatbot · Objective Metrics Guide"); canv.drawRightString(W-2*cm, H-1.3*cm, f"Page {doc.page}"); canv.restoreState()

# ─────────────────────────────────────────────────────────────────────────────
# BUILD
# ─────────────────────────────────────────────────────────────────────────────
doc = SimpleDocTemplate(OUTPUT, pagesize=A4, leftMargin=2*cm, rightMargin=2*cm, topMargin=2.2*cm, bottomMargin=2*cm)
story = [PageBreak()]  # Cover is first
story.append(Paragraph("Overview — Objective Metrics Only", ST["overview_h"]))
story.append(Paragraph("Following user feedback, the qualitative LLM Judge has been removed. Evaluation is now 100% objective, focusing on semantic truth and routing accuracy.", ST["body"]))

# Glance table
ov = [[Paragraph(h, ST["tbl_hdr"]) for h in ["Metric", "Sheet", "Weight", "Goal"]],
      ["Time Taken (s)", "Production", "—", "Fast server processing"],
      ["Context", "Both", "—", "Visible facts used for answer"],
      ["BERTScore F1 %", "Evaluation", "50 %", "Semantic truth vs context"],
      ["Source Score", "Evaluation", "25 %", "Correct system routing"],
      ["Link Validity", "Evaluation", "25 %", "Live resource check"],
      ["Accuracy %", "Evaluation", "Total", "Weighted objective score"]]
ov_t = Table(ov, colWidths=[4*cm, 3*cm, 2.5*cm, 7.5*cm])
ov_t.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,0),C["navy"]),("TEXTCOLOR",(0,0),(-1,0),colors.white),("GRID",(0,0),(-1,-1),0.3,C["divider"]),("ROWBACKGROUNDS",(0,1),(-1,-1),[C["white"],C["blue_pale"]])]))
story.append(ov_t)

# Sections
story += section_divider(1, "Performance & Transparency")
story += metric_card("Time Taken (s)", "PROD", C["blue"], "Server processing time.", ["T_end - T_start"], "Calculated on server-side.", [("0-2s","Great","green"),(">5s","Slow","red")], "Background logging ensures zero impact on user.")
story += metric_card("Context", "BOTH", C["accent"], "Raw data retrieved to answer the question.", ["KB Facts | SQL Rows | PDF Snippets"], "Passed from backend to log and test suite.", [("Present","Transparent","teal"),("Empty","Review","amber")], "This is the primary transparency tool.")

story.append(PageBreak())
story += section_divider(2, "Objective Quality Metrics")
story += metric_card("BERTScore F1 %", "EVAL", C["indigo"], "Semantic similarity between answer and context.", ["F1 (BERT-style) mapping"], "Uses all-MiniLM-L6-v2 sentence embeddings.", [("80-100%","Accurate","green"),("<50%","Suspect","red")], "Weighted at 50% of Final Accuracy.")
story += metric_card("Link Validity", "EVAL", C["amber"], "Whether links in the answer are live.", ["X/Y Valid"], "Pings URLs via HTTP HEAD/GET.", [("All Valid","Good","green"),("Partial","Stale","red")], "Weighted at 25% of Final Accuracy.")

story.append(PageBreak())
story += section_divider(3, "Final Accuracy")
story += metric_card("Accuracy %", "EVAL", C["blue"], "Total balanced quality score.", ["BERT(0.5) + Source(0.25) + Links(0.25)"], "Weighted average of all objective checks.", [("85-100%","Reliable","green"),("<65%","Fix Needed","red")], "Strictly objective; no LLM judge bias.")

doc.build(story, onFirstPage=on_page, onLaterPages=on_page)
print(f"✓ PDF generated: {OUTPUT}")
