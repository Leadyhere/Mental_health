"use client";

import { useRouter } from "next/navigation";
import { useEffect, useState } from "react";
import { ChatWindow } from "@/components/chat/ChatWindow";
import { CrisisBanner } from "@/components/crisis/CrisisBanner";
import { useChatSocket } from "@/hooks/useChatSocket";

export function SessionClient({ sessionId }: { sessionId: string }) {
  const router = useRouter();
  const [token, setToken] = useState<string | null>(null);

  useEffect(() => {
    setToken(sessionStorage.getItem(`session-token-${sessionId}`));
  }, [sessionId]);

  const { entries, pendingQuestion, crisisBanner, reportReady, connected, sendText, sendChip } =
    useChatSocket(sessionId, token ?? "");

  useEffect(() => {
    if (reportReady) {
      router.push(`/report/${sessionId}`);
    }
  }, [reportReady, router, sessionId]);

  if (!token) {
    return (
      <div className="min-h-screen flex items-center justify-center px-4 text-center">
        <p className="text-gray-500">
          No active session found for this link. Please{" "}
          <a href="/" className="text-blue-600 underline">
            start a new session
          </a>
          .
        </p>
      </div>
    );
  }

  const disabled = !connected || Boolean(crisisBanner) || reportReady;

  return (
    <div className="min-h-screen flex flex-col">
      {crisisBanner && <CrisisBanner banner={crisisBanner} />}
      <div className={`flex-1 flex flex-col max-w-2xl w-full mx-auto ${crisisBanner ? "pt-28" : ""}`}>
        <ChatWindow
          entries={entries}
          pendingQuestion={pendingQuestion}
          disabled={disabled}
          onSendText={sendText}
          onSendChip={sendChip}
        />
      </div>
    </div>
  );
}
