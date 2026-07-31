import { ReportResponse, SessionCreateResponse } from "./types";

export const API_BASE = process.env.NEXT_PUBLIC_API_BASE ?? "http://localhost:8000";
export const WS_BASE = process.env.NEXT_PUBLIC_WS_BASE ?? "ws://localhost:8000";

export async function createSession(): Promise<SessionCreateResponse> {
  const res = await fetch(`${API_BASE}/api/sessions`, { method: "POST" });
  if (!res.ok) throw new Error("failed to create session");
  return res.json();
}

export async function getReport(sessionId: string): Promise<ReportResponse> {
  const res = await fetch(`${API_BASE}/api/sessions/${sessionId}/report`);
  if (!res.ok) throw new Error("report not available yet");
  return res.json();
}

export function getReportPdfUrl(sessionId: string): string {
  return `${API_BASE}/api/sessions/${sessionId}/report?format=pdf`;
}

export async function endSession(sessionId: string): Promise<void> {
  await fetch(`${API_BASE}/api/sessions/${sessionId}/end`, { method: "POST" });
}
