import { ChatEntry } from "@/lib/types";

export function MessageBubble({ entry }: { entry: ChatEntry }) {
  const isUser = entry.role === "user";
  return (
    <div className={`flex ${isUser ? "justify-end" : "justify-start"}`}>
      <div
        className={`max-w-[75%] rounded-2xl px-4 py-2 text-sm whitespace-pre-wrap ${
          isUser
            ? "bg-blue-600 text-white rounded-br-sm"
            : "bg-gray-100 dark:bg-gray-800 text-gray-900 dark:text-gray-100 rounded-bl-sm"
        }`}
      >
        {entry.text}
      </div>
    </div>
  );
}
