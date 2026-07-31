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
    from fpdf import FPDF

    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 10, "Session Summary", ln=True)

    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 10, f"Risk Level: {report.get('risk_level') or 'N/A'}", ln=True)

    pdf.set_font("Helvetica", "", 11)
    if report.get("narrative_highlights"):
        pdf.ln(4)
        pdf.set_font("Helvetica", "B", 12)
        pdf.cell(0, 8, "Narrative Highlights", ln=True)
        pdf.set_font("Helvetica", "", 11)
        for key, value in report["narrative_highlights"].items():
            pdf.multi_cell(0, 7, f"{key}: {value}")

    if report.get("structured_answers"):
        pdf.ln(4)
        pdf.set_font("Helvetica", "B", 12)
        pdf.cell(0, 8, "Structured Answers", ln=True)
        pdf.set_font("Helvetica", "", 11)
        for qa in report["structured_answers"]:
            pdf.multi_cell(0, 7, f"{qa['question']} -> {qa['answer']}")

    if report.get("suggested_next_steps"):
        pdf.ln(4)
        pdf.set_font("Helvetica", "B", 12)
        pdf.cell(0, 8, "Suggested Next Steps", ln=True)
        pdf.set_font("Helvetica", "", 11)
        pdf.multi_cell(0, 7, report["suggested_next_steps"])

    return bytes(pdf.output())
