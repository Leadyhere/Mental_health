from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Literal, Optional
from uuid import UUID, uuid4

from pydantic import BaseModel, Field

RiskLevel = Literal["N/A-Feeling Fine", "Low", "Mild", "Moderate", "Severe", "Emergency"]


class InputType(str, Enum):
    text = "text"
    audio = "audio"  # stubbed this phase, no audio pipeline wired up


class ConversationPhase(str, Enum):
    orientation = "orientation"
    narrative = "narrative"
    narrative_analysis = "narrative_analysis"
    structured = "structured"
    closing = "closing"
    emergency = "emergency"


class Utterance(BaseModel):
    turn_id: int
    role: Literal["user", "assistant"]
    text: str
    input_type: InputType = InputType.text
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class NarrativeAnalysis(BaseModel):
    """Internal state only -- never shown to the user."""

    key_events: list[str] = Field(default_factory=list)
    timeline: Optional[str] = None
    emotions: list[str] = Field(default_factory=list)
    coping_signals: list[str] = Field(default_factory=list)
    support_signals: list[str] = Field(default_factory=list)
    functional_impact: Optional[str] = None
    risk_language_flags: list[str] = Field(default_factory=list)
    answered_questions: dict[str, bool] = Field(default_factory=dict)


class StructuredAnswer(BaseModel):
    question_id: str
    question_text: str
    answer_text: Optional[str] = None
    chip_selected: Optional[str] = None
    asked_verbatim: bool = False


class SubLabels(BaseModel):
    suicide_risk: str
    functional_impairment: str
    support_level: str
    help_readiness: str


class RiskAssessment(BaseModel):
    risk_level: RiskLevel
    sub_labels: SubLabels
    confidence: float
    source: Literal["llm", "inhouse"]
    rationale: Optional[str] = None
    is_reviewed: bool = False
    reviewer_verdict: Optional[str] = None


class SessionState(BaseModel):
    session_id: UUID = Field(default_factory=uuid4)
    phase: ConversationPhase = ConversationPhase.orientation
    started_at: datetime = Field(default_factory=datetime.utcnow)
    utterances: list[Utterance] = Field(default_factory=list)
    narrative_analysis: Optional[NarrativeAnalysis] = None
    structured_answers: list[StructuredAnswer] = Field(default_factory=list)
    current_risk_assessment: Optional[RiskAssessment] = None
    emergency_triggered: bool = False
    structured_question_count: int = 0
    self_assessment: Optional[dict[str, Any]] = None
    narrative_completion_check_asked: bool = False
    pending_question_id: Optional[str] = None


class AnonymizedSessionRecord(BaseModel):
    session_id: UUID
    risk_level: RiskLevel
    sub_labels: SubLabels
    confidence: float
    model_version: str
    embeddings_ref: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    is_reviewed: bool = False
    reviewer_verdict: Optional[str] = None


class TrainingTriple(BaseModel):
    session_id: UUID
    narrative_features: dict[str, Any]
    structured_answers: list[StructuredAnswer]
    llm_risk_level: RiskLevel
    llm_sub_labels: SubLabels
    llm_rationale: Optional[str] = None
    inhouse_risk_level: Optional[str] = None
    inhouse_confidence: Optional[float] = None
    is_reviewed: bool = False
    reviewer_verdict: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)


# --- WebSocket wire messages ---

class ClientMessage(BaseModel):
    type: Literal["user_text", "chip_select", "request_status", "done_sharing"]
    payload: dict[str, Any] = Field(default_factory=dict)


class ServerMessage(BaseModel):
    type: Literal["assistant_text", "chip_options", "phase_change", "crisis_banner", "report_ready"]
    payload: dict[str, Any] = Field(default_factory=dict)
