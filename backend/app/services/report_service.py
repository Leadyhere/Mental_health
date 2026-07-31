"""Builds the post-session structured summary: narrative highlights,
structured answers, risk level, and suggested next steps (verbatim copy
from the dataset's `Risk Framework` sheet, see docs/data_notes.md).
"""
from __future__ import annotations

import io

from app.models.schemas import SessionState

NEXT_STEPS_BY_RISK_LEVEL = {
    "N/A-Feeling Fine": "Glad to hear things feel okay right now. No action needed -- come back any time.",
    "Low": (
        "What you're describing sounds difficult but not unsafe. Try one supportive conversation, "
        "light self-monitoring (sleep/appetite/mood), and 2-3 small stabilizers this week such as "
        "hydration, a walk, or a simple routine."
    ),
    "Mild": (
        "Would it help to try a small plan for the next 7 days? Pick one daily stabilizer (sleep window, "
        "movement, social contact), speak with a trusted person or low-barrier counselor, and use a "
        "professional directory if needed."
    ),
    "Moderate": (
        "It would be a good idea to talk to a counsellor or psychologist within 1-2 weeks. Use this "
        "summary as a starter script, reduce substances, keep sleep regular, and plan one manageable "
        "task per day."
    ),
    "Severe": (
        "A professional evaluation is recommended soon (ideally within 24-72 hours if possible). Do not "
        "stay alone if you feel unsafe, involve a trusted adult/person, and seek a psychologist or "
        "psychiatrist, especially if functioning is strongly affected or symptoms feel out of control."
    ),
    "Emergency": (
        "I'm concerned about your safety right now. Contact a trusted adult/person immediately, call a "
        "crisis helpline or emergency service, or go to the nearest ER. A safety plan can be made with a "
        "professional, including warning signs, coping steps, supportive contacts, and reducing access to means."
    ),
}


def build_report(state: SessionState) -> dict:
    assessment = state.current_risk_assessment
    narrative_highlights = None
    if state.narrative_analysis:
        narrative_highlights = {
            "key_events": state.narrative_analysis.key_events,
            "emotions": state.narrative_analysis.emotions,
            "coping_signals": state.narrative_analysis.coping_signals,
            "support_signals": state.narrative_analysis.support_signals,
            "functional_impact": state.narrative_analysis.functional_impact,
        }

    return {
        "session_id": str(state.session_id),
        "risk_level": assessment.risk_level if assessment else None,
        "rationale": assessment.rationale if assessment else None,
        "sub_labels": assessment.sub_labels.model_dump() if assessment else None,
        "narrative_highlights": narrative_highlights,
        "structured_answers": [
            {"question": a.question_text, "answer": a.chip_selected or a.answer_text}
            for a in state.structured_answers
        ],
        "suggested_next_steps": NEXT_STEPS_BY_RISK_LEVEL.get(assessment.risk_level) if assessment else None,
    }


def build_report_pdf(report: dict) -> bytes:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.platypus import HRFlowable, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        rightMargin=36,
        leftMargin=36,
        topMargin=36,
        bottomMargin=36,
    )

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "DocTitle",
        parent=styles["Heading1"],
        fontName="Helvetica-Bold",
        fontSize=18,
        leading=22,
        textColor=colors.HexColor("#1e293b"),
    )
    subtitle_style = ParagraphStyle(
        "SubTitle",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=10,
        leading=13,
        textColor=colors.HexColor("#64748b"),
    )
    section_heading = ParagraphStyle(
        "SectionHeading",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=12,
        leading=15,
        textColor=colors.HexColor("#0f172a"),
        spaceBefore=10,
        spaceAfter=4,
    )
    body_style = ParagraphStyle(
        "BodyTextCustom",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=10,
        leading=14,
        textColor=colors.HexColor("#334155"),
    )

    elements = []

    # Title & FHIR Header
    elements.append(Paragraph("Clinical Triage Summary — HL7 FHIR RiskAssessment", title_style))
    elements.append(Paragraph(f"FHIR Resource: RiskAssessment | Subject Session ID: {report.get('session_id')} | Status: final", subtitle_style))
    elements.append(Spacer(1, 10))
    elements.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#cbd5e1"), spaceAfter=10))

    # Risk Tier Heatmap Badge
    risk_level = report.get("risk_level") or "N/A"
    risk_colors = {
        "Emergency": "#dc2626",
        "Severe": "#ea580c",
        "Moderate": "#d97706",
        "Mild": "#ca8a04",
        "Low": "#2563eb",
        "N/A-Feeling Fine": "#16a34a",
    }
    badge_bg = risk_colors.get(risk_level, "#475569")

    heatmap_data = [
        [
            Paragraph(f"<font color='white'><b>RISK TIER: {risk_level.upper()}</b></font>", ParagraphStyle("Badge", parent=body_style, fontSize=11, leading=14)),
            Paragraph(f"<b>Rationale:</b> {report.get('rationale') or 'N/A'}", body_style),
        ]
    ]
    t_heatmap = Table(heatmap_data, colWidths=[160, 380])
    t_heatmap.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (0, 0), colors.HexColor(badge_bg)),
            ("ALIGN", (0, 0), (0, 0), "CENTER"),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("BACKGROUND", (1, 0), (1, 0), colors.HexColor("#f8fafc")),
            ("BOX", (0, 0), (-1, -1), 1, colors.HexColor("#cbd5e1")),
            ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
            ("TOPPADDING", (0, 0), (-1, -1), 8),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
            ("LEFTPADDING", (0, 0), (-1, -1), 8),
            ("RIGHTPADDING", (0, 0), (-1, -1), 8),
        ])
    )
    elements.append(t_heatmap)
    elements.append(Spacer(1, 12))

    # Clinical Indicators (Sub-labels)
    if report.get("sub_labels"):
        elements.append(Paragraph("Clinical Indicators (FHIR Observations)", section_heading))
        sub_data = [["Indicator", "Clinical Rating"]]
        for k, v in report["sub_labels"].items():
            sub_data.append([k.replace("_", " ").title(), str(v)])
        t_sub = Table(sub_data, colWidths=[200, 340])
        t_sub.setStyle(
            TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#f1f5f9")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#0f172a")),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("BOTTOMPADDING", (0, 0), (-1, 0), 6),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#e2e8f0")),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ])
        )
        elements.append(t_sub)
        elements.append(Spacer(1, 12))

    # Narrative Highlights
    if report.get("narrative_highlights"):
        elements.append(Paragraph("Narrative Synthesis Highlights", section_heading))
        nh_data = []
        for k, v in report["narrative_highlights"].items():
            nh_data.append([Paragraph(f"<b>{k.replace('_', ' ').title()}:</b>", body_style), Paragraph(str(v), body_style)])
        t_nh = Table(nh_data, colWidths=[140, 400])
        t_nh.setStyle(
            TableStyle([
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ])
        )
        elements.append(t_nh)
        elements.append(Spacer(1, 12))

    # Structured Answers
    if report.get("structured_answers"):
        elements.append(Paragraph("Structured Screening Responses", section_heading))
        qa_data = [["Question Prompt", "User Selected Response"]]
        for qa in report["structured_answers"]:
            qa_data.append([Paragraph(qa["question"], body_style), Paragraph(qa["answer"], body_style)])
        t_qa = Table(qa_data, colWidths=[300, 240])
        t_qa.setStyle(
            TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#f1f5f9")),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#e2e8f0")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ])
        )
        elements.append(t_qa)
        elements.append(Spacer(1, 12))

    # Recommended Next Steps
    if report.get("suggested_next_steps"):
        elements.append(Paragraph("Recommended Clinical Interventions & Next Steps", section_heading))
        elements.append(Paragraph(report["suggested_next_steps"], body_style))

    doc.build(elements)
    return buffer.getvalue()
