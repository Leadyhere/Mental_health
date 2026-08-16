from datetime import datetime, timezone


class ReportBuilder:
    def build(self, session: dict) -> tuple[str, dict[str, object]]:
        slots = session["filled_slots"]
        risk = session["highest_risk_level"]
        concerns = {
            "trigger": slots.get("trigger"),
            "duration": slots.get("duration"),
            "impact": slots.get("impact"),
            "emotions": slots.get("emotions", []),
            "coping": slots.get("coping"),
            "support": slots.get("support"),
        }
        lines = [
            "MINDTRIAGE CONVERSATION SUMMARY",
            f"Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
            f"Highest observed risk category: {risk}",
            "",
            "This is a non-diagnostic conversation summary, not a medical assessment.",
            "",
            "Concerns explored:",
            f"- Main area: {concerns['trigger'] or 'Not specified'}",
            f"- Duration: {concerns['duration'] or 'Not specified'}",
            f"- Daily impact: {concerns['impact'] or 'Not specified'}",
            f"- Emotions: {', '.join(concerns['emotions']) or 'Not specified'}",
            f"- Coping: {concerns['coping'] or 'Not specified'}",
            f"- Support: {concerns['support'] or 'Not specified'}",
            "",
            "Suggested next step:",
        ]
        if "Level 5" in risk:
            lines.append("Contact emergency services at 112 or Tele-MANAS at 14416 now.")
        elif "Level 4" in risk:
            lines.append("Seek prompt support from a qualified mental-health professional.")
        elif "Level 3" in risk:
            lines.append("Consider arranging a mental-health consultation soon.")
        else:
            lines.append("Continue monitoring how you feel and consider speaking with someone you trust.")
        lines.extend(
            [
                "",
                "Privacy: active session data is deleted when the session ends or expires.",
            ]
        )
        return "\n".join(lines), concerns
