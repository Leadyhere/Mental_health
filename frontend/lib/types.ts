export type ServerMessageType =
  | "assistant_text"
  | "chip_options"
  | "phase_change"
  | "crisis_banner"
  | "report_ready";

export type ClientMessageType = "user_text" | "chip_select" | "request_status" | "done_sharing";

export interface ServerMessage {
  type: ServerMessageType;
  payload: Record<string, unknown>;
}

export interface ClientMessage {
  type: ClientMessageType;
  payload: Record<string, unknown>;
}

export interface ChipOptionsPayload {
  question_id: string;
  prompt: string;
  options: string[];
  allow_free_text: boolean;
  asked_verbatim: boolean;
}

export interface CrisisResource {
  name: string;
  phone: string;
  description: string;
}

export interface CrisisBannerPayload {
  message: string;
  resources: CrisisResource[];
}

export type ConversationPhase =
  | "orientation"
  | "narrative"
  | "narrative_analysis"
  | "structured"
  | "closing"
  | "emergency";

export interface ChatEntry {
  id: string;
  role: "user" | "assistant";
  text: string;
}

export interface SessionCreateResponse {
  session_id: string;
  token: string;
  orientation_message: string;
}

export interface ReportResponse {
  session_id: string;
  risk_level: string | null;
  rationale: string | null;
  sub_labels: Record<string, string> | null;
  narrative_highlights: Record<string, unknown> | null;
  structured_answers: { question: string; answer: string }[];
  suggested_next_steps: string | null;
}
