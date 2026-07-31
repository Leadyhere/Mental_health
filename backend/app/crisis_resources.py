"""India-specific crisis resources, surfaced unconditionally whenever the
Emergency gate triggers. Single source of truth -- imported by both
emergency_gate.py and the /api/crisis-resources endpoint so the frontend
never hardcodes these numbers separately.
"""

CRISIS_RESOURCES = [
    {"name": "Tele-MANAS", "phone": "14416", "description": "24/7 national mental health helpline (India)"},
    {"name": "KIRAN", "phone": "1800-599-0019", "description": "24/7 national mental health rehabilitation helpline (India)"},
    {"name": "iCALL", "phone": "9152987821", "description": "Psychosocial helpline by TISS (India)"},
]

EMERGENCY_MESSAGE = (
    "I'm concerned about your safety right now. Contact a trusted adult/person "
    "immediately, call a crisis helpline or emergency service, or go to the "
    "nearest ER. You don't have to face this alone."
)
