"use client";

import { useRouter } from "next/navigation";
import { useEffect, useState } from "react";
import { AuthModal } from "@/components/auth/AuthModal";
import { ChatWindow } from "@/components/chat/ChatWindow";
import { CrisisBanner } from "@/components/crisis/CrisisBanner";
import { useChatSocket } from "@/hooks/useChatSocket";

export function SessionClient({ sessionId }: { sessionId: string }) {
  const router = useRouter();
  const [token, setToken] = useState<string | null>(null);
  const [isPanicMode, setIsPanicMode] = useState(false);
  const [isPinModalOpen, setIsPinModalOpen] = useState(false);
  const [isLocked, setIsLocked] = useState(false);

  useEffect(() => {
    setToken(sessionStorage.getItem(`session-token-${sessionId}`));
  }, [sessionId]);

  const { entries, phase, pendingQuestion, crisisBanner, reportReady, connected, sendText, sendChip, sendDoneSharing } =
    useChatSocket(sessionId, token ?? "");

  // ESC key listener for Panic Mode
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === "Escape") {
        setIsPanicMode(true);
        document.title = "Google";
      }
    };
    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, []);

  const triggerPanic = () => {
    setIsPanicMode(true);
    document.title = "Google";
  };

  useEffect(() => {
    if (reportReady) {
      router.push(`/report/${sessionId}`);
    }
  }, [reportReady, router, sessionId]);

  if (isPanicMode) {
    return (
      <div className="min-h-screen bg-white text-gray-900 font-sans p-6">
        <div className="max-w-2xl mx-auto space-y-6 pt-8">
          <div className="flex items-center gap-4">
            <span className="text-3xl font-bold tracking-tight text-blue-600">Google</span>
            <input
              type="text"
              defaultValue="How to make sourdough bread from scratch"
              className="flex-1 rounded-full border border-gray-300 px-4 py-2 text-sm shadow-sm focus:outline-none"
            />
          </div>
          <div className="border-b border-gray-200 pb-2 text-xs text-gray-500 flex gap-6">
            <span className="text-blue-600 font-semibold border-b-2 border-blue-600 pb-2">All</span>
            <span>Images</span>
            <span>Videos</span>
            <span>News</span>
            <span>Shopping</span>
          </div>
          <div className="space-y-4 pt-2">
            <div>
              <p className="text-xs text-gray-600">https://www.kingarthurbaking.com › recipes › sourdough</p>
              <h3 className="text-lg text-blue-800 hover:underline cursor-pointer">Sourdough Bread Recipe | King Arthur Baking</h3>
              <p className="text-sm text-gray-700">Learn how to make a classic sourdough bread with crisp crust, open crumb, and chewy texture using a active sourdough starter.</p>
            </div>
            <div>
              <p className="text-xs text-gray-600">https://www.thekitchn.com › sourdough-bread-recipe</p>
              <h3 className="text-lg text-blue-800 hover:underline cursor-pointer">How To Make Sourdough Bread: The Complete Guide</h3>
              <p className="text-sm text-gray-700">A step-by-step master guide with pictures showing timing, stretching, folding, shaping, and baking beautiful sourdough loaves.</p>
            </div>
          </div>
        </div>
      </div>
    );
  }

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

  if (isLocked) {
    return (
      <div className="min-h-screen flex flex-col items-center justify-center bg-gray-100 dark:bg-gray-950 p-4">
        <div className="text-center space-y-4">
          <div className="mx-auto w-16 h-16 rounded-full bg-blue-100 dark:bg-blue-900/50 flex items-center justify-center text-blue-600 text-2xl">
            🔒
          </div>
          <h2 className="text-2xl font-bold text-gray-900 dark:text-white">Session Locked</h2>
          <p className="text-sm text-gray-500">Enter your PIN to unlock and resume your session.</p>
          <button
            onClick={() => setIsPinModalOpen(true)}
            className="px-6 py-2.5 rounded-xl bg-blue-600 text-white font-medium hover:bg-blue-700 transition-colors"
          >
            Unlock Session
          </button>
        </div>
        <AuthModal
          isOpen={isPinModalOpen}
          onUnlock={() => {
            setIsLocked(false);
            setIsPinModalOpen(false);
          }}
          onClose={() => setIsPinModalOpen(false)}
        />
      </div>
    );
  }

  const disabled = !connected || Boolean(crisisBanner) || reportReady;

  return (
    <div className="min-h-screen flex flex-col bg-white dark:bg-gray-950">
      {crisisBanner && <CrisisBanner banner={crisisBanner} />}
      <div className={`flex-1 flex flex-col max-w-2xl w-full mx-auto ${crisisBanner ? "pt-28" : ""}`}>
        <ChatWindow
          entries={entries}
          phase={phase}
          pendingQuestion={pendingQuestion}
          disabled={disabled}
          onSendText={sendText}
          onSendChip={sendChip}
          onDoneSharing={sendDoneSharing}
          onPanicTrigger={triggerPanic}
          onLockTrigger={() => {
            setIsLocked(true);
            setIsPinModalOpen(true);
          }}
        />
      </div>
      <AuthModal
        isOpen={isPinModalOpen}
        onUnlock={() => {
          setIsLocked(false);
          setIsPinModalOpen(false);
        }}
        onClose={() => setIsPinModalOpen(false)}
      />
    </div>
  );
}
