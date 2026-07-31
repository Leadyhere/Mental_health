"use client";

import { useRouter } from "next/navigation";
import { useState } from "react";
import { createSession } from "@/lib/apiClient";

export default function Home() {
  const router = useRouter();
  const [starting, setStarting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const start = async () => {
    setStarting(true);
    setError(null);
    try {
      const { session_id, token } = await createSession();
      sessionStorage.setItem(`session-token-${session_id}`, token);
      router.push(`/session/${session_id}`);
    } catch {
      setError("Could not reach the server. Is the backend running on localhost:8000?");
      setStarting(false);
    }
  };

  return (
    <div className="min-h-screen flex items-center justify-center px-4">
      <div className="max-w-md text-center space-y-6">
        <h1 className="text-2xl font-semibold">You&apos;re not alone in this.</h1>
        <p className="text-gray-600 dark:text-gray-400">
          This is a private space to talk through what&apos;s been weighing on you. It&apos;s not
          therapy, and it won&apos;t diagnose you -- it&apos;s here to listen and help you sort
          things out.
        </p>
        <button
          onClick={start}
          disabled={starting}
          className="rounded-xl bg-blue-600 text-white px-6 py-3 font-medium disabled:opacity-50"
        >
          {starting ? "Starting..." : "Begin"}
        </button>
        {error && <p className="text-sm text-red-600">{error}</p>}
      </div>
    </div>
  );
}
