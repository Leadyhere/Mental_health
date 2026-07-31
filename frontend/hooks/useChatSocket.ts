"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { WS_BASE } from "@/lib/apiClient";
import {
  ChatEntry,
  ChipOptionsPayload,
  ClientMessage,
  ConversationPhase,
  CrisisBannerPayload,
  ServerMessage,
} from "@/lib/types";

interface UseChatSocketResult {
  entries: ChatEntry[];
  phase: ConversationPhase;
  pendingQuestion: ChipOptionsPayload | null;
  crisisBanner: CrisisBannerPayload | null;
  reportReady: boolean;
  connected: boolean;
  sendText: (text: string) => void;
  sendChip: (value: string) => void;
  sendDoneSharing: () => void;
}

let entryCounter = 0;
const nextId = () => `entry-${Date.now()}-${entryCounter++}`;

export function useChatSocket(sessionId: string, token: string): UseChatSocketResult {
  const [entries, setEntries] = useState<ChatEntry[]>([]);
  const [phase, setPhase] = useState<ConversationPhase>("narrative");
  const [pendingQuestion, setPendingQuestion] = useState<ChipOptionsPayload | null>(null);
  const [crisisBanner, setCrisisBanner] = useState<CrisisBannerPayload | null>(null);
  const [reportReady, setReportReady] = useState(false);
  const [connected, setConnected] = useState(false);
  const wsRef = useRef<WebSocket | null>(null);

  useEffect(() => {
    if (!sessionId || !token) return;
    const ws = new WebSocket(`${WS_BASE}/api/sessions/${sessionId}/chat?token=${token}`);
    wsRef.current = ws;

    ws.onopen = () => setConnected(true);
    ws.onclose = () => setConnected(false);
    ws.onmessage = (event) => {
      const message: ServerMessage = JSON.parse(event.data);
      switch (message.type) {
        case "assistant_text":
          setEntries((prev) => [
            ...prev,
            { id: nextId(), role: "assistant", text: String(message.payload.text) },
          ]);
          setPendingQuestion(null);
          break;
        case "chip_options":
          setPendingQuestion(message.payload as unknown as ChipOptionsPayload);
          setEntries((prev) => [
            ...prev,
            { id: nextId(), role: "assistant", text: String(message.payload.prompt) },
          ]);
          break;
        case "phase_change":
          setPhase(message.payload.phase as ConversationPhase);
          break;
        case "crisis_banner":
          setCrisisBanner(message.payload as unknown as CrisisBannerPayload);
          break;
        case "report_ready":
          setReportReady(true);
          break;
      }
    };

    return () => {
      ws.close();
    };
  }, [sessionId, token]);

  const send = useCallback((message: ClientMessage) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify(message));
    }
  }, []);

  const sendText = useCallback(
    (text: string) => {
      if (!text.trim()) return;
      setEntries((prev) => [...prev, { id: nextId(), role: "user", text }]);
      send({ type: "user_text", payload: { text } });
    },
    [send]
  );

  const sendChip = useCallback(
    (value: string) => {
      setEntries((prev) => [...prev, { id: nextId(), role: "user", text: value }]);
      setPendingQuestion(null);
      send({ type: "chip_select", payload: { chip_value: value } });
    },
    [send]
  );

  const sendDoneSharing = useCallback(() => {
    send({ type: "done_sharing", payload: { done_sharing: true } });
  }, [send]);

  return { entries, phase, pendingQuestion, crisisBanner, reportReady, connected, sendText, sendChip, sendDoneSharing };
}
