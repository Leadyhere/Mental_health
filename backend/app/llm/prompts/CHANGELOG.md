# Prompt changelog

## v1 (initial)
- orientation, listening_ack, completion_check, transition_to_structured,
  narrative_analysis, risk_classifier, safety_filter_check.
- Q9 (passive SI) / Q10 (active SI) verbatim strings are NOT prompt
  templates -- they live as Python constants in
  `backend/app/conversation/question_bank.py` and must never be routed
  through the LLM for rephrasing.
