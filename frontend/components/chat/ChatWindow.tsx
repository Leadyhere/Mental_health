"use client";

import { useEffect, useRef } from "react";
import { ChatEntry, ChipOptionsPayload } from "@/lib/types";
import { MessageBubble } from "./MessageBubble";
import { NarrativeInput } from "./NarrativeInput";
import { PromptChips } from "./PromptChips";
import { RiskQuestionCard } from "./RiskQuestionCard";

export function ChatWindow({
  entries,
  pendingQuestion,
  disabled,
  onSendText,
  onSendChip,
}: {
  entries: ChatEntry[];
  pendingQuestion: ChipOptionsPayload | null;
  disabled: boolean;
  onSendText: (text: string) => void;
  onSendChip: (value: string) => void;
}) {
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [entries, pendingQuestion]);

  return (
    <div className="flex flex-col h-full">
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
        <NarrativeInput onSubmit={onSendText} disabled={disabled} />
      </div>
    </div>
  );
}
