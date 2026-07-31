---
phase: narrative_analysis
returns: json
schema: NarrativeAnalysis
---
You are analyzing a free-form story someone told a mental-health triage tool about what's been weighing on them. Extract structured signal from it. Never diagnose. Do not invent details that aren't implied by the text.

The full narrative (all turns concatenated):
"""
{narrative_text}
"""

The assessment question bank has these question ids, each about a specific topic:
- emotional_clarification: what feeling words / emotional states did they name or clearly imply?
- coping_behaviors: what coping behaviors did they mention?
- support_person: did they mention anyone they talk to, or lack thereof?
- distress_intensity: did they give any indication of how intense this feels?
- feels_safe: did they say anything about feeling unsafe or safe right now? (Do NOT infer this from tone alone -- only mark answered if genuinely explicit.)
- meaning_readiness: did they say what they'd hope would be different?
- open_to_support: did they express any openness (or reluctance) to getting help?

Never mark passive_si or active_si as "answered" here -- those must always be asked directly, verbatim, regardless of what the narrative implies.

Return ONLY a JSON object with this exact shape:
{{
  "key_events": ["..."],
  "timeline": "string or null",
  "emotions": ["..."],
  "coping_signals": ["..."],
  "support_signals": ["..."],
  "functional_impact": "string or null",
  "risk_language_flags": ["any risk-relevant phrases verbatim, for internal safety review only"],
  "answered_questions": {{"emotional_clarification": true/false, "coping_behaviors": true/false, "support_person": true/false, "distress_intensity": true/false, "feels_safe": true/false, "meaning_readiness": true/false, "open_to_support": true/false}}
}}
