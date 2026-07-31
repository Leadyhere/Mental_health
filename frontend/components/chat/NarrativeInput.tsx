"use client";

import { useState } from "react";

export function NarrativeInput({
  onSubmit,
  onDoneSharing,
  disabled,
  placeholder,
  showDoneButton,
}: {
  onSubmit: (text: string) => void;
  onDoneSharing?: () => void;
  disabled?: boolean;
  placeholder?: string;
  showDoneButton?: boolean;
}) {
  const [text, setText] = useState("");

  const submit = () => {
    if (!text.trim() || disabled) return;
    onSubmit(text);
    setText("");
  };

  return (
    <div className="flex flex-col gap-2">
      {showDoneButton && onDoneSharing && (
        <div className="flex justify-end">
          <button
            onClick={onDoneSharing}
            disabled={disabled}
            className="text-xs font-medium text-blue-600 hover:text-blue-700 dark:text-blue-400 hover:underline flex items-center gap-1 py-1 px-2 rounded-lg bg-blue-50 dark:bg-blue-950/40 border border-blue-200 dark:border-blue-800/60"
          >
            [ Done sharing for now → ]
          </button>
        </div>
      )}
      <div className="flex gap-2 items-end">
        <textarea
          className="flex-1 resize-none rounded-xl border border-gray-300 dark:border-gray-700 bg-white dark:bg-gray-900 px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
          rows={2}
          value={text}
          placeholder={placeholder ?? "Type in your own words..."}
          disabled={disabled}
          onChange={(e) => setText(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter" && !e.shiftKey) {
              e.preventDefault();
              submit();
            }
          }}
        />
        <button
          onClick={submit}
          disabled={disabled}
          className="rounded-xl bg-blue-600 text-white px-4 py-2 text-sm font-medium disabled:opacity-40 hover:bg-blue-700 transition-colors"
        >
          Send
        </button>
      </div>
    </div>
  );
}
