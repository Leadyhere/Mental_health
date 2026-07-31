"use client";

import { useState } from "react";

export function NarrativeInput({
  onSubmit,
  disabled,
  placeholder,
}: {
  onSubmit: (text: string) => void;
  disabled?: boolean;
  placeholder?: string;
}) {
  const [text, setText] = useState("");

  const submit = () => {
    if (!text.trim() || disabled) return;
    onSubmit(text);
    setText("");
  };

  return (
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
        className="rounded-xl bg-blue-600 text-white px-4 py-2 text-sm font-medium disabled:opacity-40"
      >
        Send
      </button>
    </div>
  );
}
