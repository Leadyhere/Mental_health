from pydantic import BaseModel, Field


class StartSessionRequest(BaseModel):
    training_consent: bool = False
    locale: str = Field(default="en-IN", max_length=20)


class MessageRequest(BaseModel):
    session_id: str = Field(min_length=8, max_length=80)
    user_input: str = Field(min_length=1, max_length=4000)


class EndSessionRequest(BaseModel):
    session_id: str = Field(min_length=8, max_length=80)


class ChatResponse(BaseModel):
    session_id: str
    bot_reply: str
    risk_level: str
    highest_risk_level: str
    is_emergency: bool
    emergency_contacts: dict[str, str]
    extracted_emotions: list[str]
    target_slot: str
    model_source: str


class SummaryResponse(BaseModel):
    session_id: str
    report: str
    risk_level: str
    concerns: dict[str, object]
