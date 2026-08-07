"use client";

import { useEffect, useRef } from "react";
import { ChatEntry, ChipOptionsPayload } from "@/lib/types";
import { MessageBubble } from "./MessageBubble";
import { NarrativeInput } from "./NarrativeInput";
import { PromptChips } from "./PromptChips";
import { RiskQuestionCard } from "./RiskQuestionCard";

export function ChatWindow({
  entries,
  phase,
  pendingQuestion,
  disabled,
  onSendText,
  onSendChip,
  onDoneSharing,
  onPanicTrigger,
  onLockTrigger,
}: {
  entries: ChatEntry[];
  phase?: string;
  pendingQuestion: ChipOptionsPayload | null;
  disabled: boolean;
  onSendText: (text: string) => void;
  onSendChip: (value: string) => void;
  onDoneSharing?: () => void;
  onPanicTrigger?: () => void;
  onLockTrigger?: () => void;
}) {
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [entries, pendingQuestion]);

  return (
    <div className="flex flex-col h-full">
      {/* Header bar with Panic & Lock controls */}
      <div className="flex items-center justify-between px-4 py-2 bg-gray-50 dark:bg-gray-900 border-b border-gray-200 dark:border-gray-800 text-xs text-gray-500">
        <div className="flex items-center gap-2">
          <span className="w-2 h-2 rounded-full bg-emerald-500 animate-pulse" />
          <span>Session Active</span>
        </div>
        <div className="flex items-center gap-2">
          {onLockTrigger && (
            <button
              onClick={onLockTrigger}
              className="px-2.5 py-1 rounded-lg bg-gray-200 dark:bg-gray-800 hover:bg-gray-300 dark:hover:bg-gray-700 font-medium text-gray-700 dark:text-gray-300 transition-colors flex items-center gap-1"
            >
              🔒 Lock PIN
            </button>
          )}
          {onPanicTrigger && (
            <button
              onClick={onPanicTrigger}
              className="px-2.5 py-1 rounded-lg bg-red-600 hover:bg-red-700 font-bold text-white transition-colors shadow-sm flex items-center gap-1"
              title="Press ESC anytime for quick exit"
            >
              ⚡ Quick Exit (ESC)
            </button>
          )}
        </div>
      </div>

      <div className="flex-1 overflow-y-auto space-y-3 px-4 py-4">
        {entries.map((entry) => (
          <MessageBubble key={entry.id} entry={entry} />
        ))}
        <div ref={bottomRef} />
      </div>

      <div className="border-t border-gray-200 dark:border-gray-800 p-4 space-y-3">
        {pendingQuestion && pendingQuestion.asked_verbatim && (
          <RiskQuestionCard
            prompt={pendingQuestion.prompt}
            options={pendingQuestion.options}
            onSelect={onSendChip}
            disabled={disabled}
          />
        )}
        {pendingQuestion && !pendingQuestion.asked_verbatim && (
          <PromptChips options={pendingQuestion.options} onSelect={onSendChip} disabled={disabled} />
        )}
        {/* Free text is always available -- chips are supplementary, never a forced choice. */}
        <NarrativeInput
          onSubmit={onSendText}
          onDoneSharing={onDoneSharing}
          disabled={disabled}
          showDoneButton={phase === "narrative" || !phase}
        />
      </div>
    </div>
  );
}
