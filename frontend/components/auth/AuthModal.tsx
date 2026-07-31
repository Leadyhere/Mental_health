"use client";

import { useState } from "react";

interface AuthModalProps {
  isOpen: boolean;
  onUnlock: () => void;
  onClose: () => void;
}

export function AuthModal({ isOpen, onUnlock, onClose }: AuthModalProps) {
  const [pin, setPin] = useState("");
  const [error, setError] = useState(false);

  if (!isOpen) return null;

  const handleDigit = (digit: string) => {
    if (pin.length < 4) {
      const newPin = pin + digit;
      setPin(newPin);
      if (newPin.length === 4) {
        // Unlock on 4 digits
        onUnlock();
        setPin("");
        setError(false);
      }
    }
  };

  const handleClear = () => {
    setPin("");
    setError(false);
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm p-4">
      <div className="w-full max-w-sm rounded-2xl bg-white dark:bg-gray-900 p-6 shadow-2xl border border-gray-200 dark:border-gray-800 text-center">
        <div className="mx-auto w-12 h-12 rounded-full bg-blue-100 dark:bg-blue-900/50 flex items-center justify-center mb-4 text-blue-600 dark:text-blue-400">
          <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 15v2m-6 4h12a2 2 0 002-2v-6a2 2 0 00-2-2H6a2 2 0 00-2 2v6a2 2 0 002 2zm10-10V7a4 4 0 00-8 0v4h8z" />
          </svg>
        </div>
        <h2 className="text-xl font-bold text-gray-900 dark:text-white mb-1">Session PIN Lock</h2>
        <p className="text-xs text-gray-500 dark:text-gray-400 mb-6">Enter any 4-digit PIN to secure or resume your confidential session.</p>

        {/* PIN display */}
        <div className="flex justify-center gap-3 mb-6">
          {[0, 1, 2, 3].map((idx) => (
            <div
              key={idx}
              className={`w-4 h-4 rounded-full border-2 transition-all ${
                pin.length > idx
                  ? "bg-blue-600 border-blue-600 scale-110"
                  : "border-gray-300 dark:border-gray-700"
              }`}
            />
          ))}
        </div>

        {/* Keypad */}
        <div className="grid grid-cols-3 gap-3 mb-4">
          {["1", "2", "3", "4", "5", "6", "7", "8", "9"].map((num) => (
            <button
              key={num}
              onClick={() => handleDigit(num)}
              className="h-12 rounded-xl bg-gray-100 dark:bg-gray-800 hover:bg-gray-200 dark:hover:bg-gray-700 text-lg font-semibold text-gray-800 dark:text-gray-200 transition-colors"
            >
              {num}
            </button>
          ))}
          <button
            onClick={handleClear}
            className="h-12 rounded-xl bg-gray-100 dark:bg-gray-800 hover:bg-gray-200 dark:hover:bg-gray-700 text-xs font-semibold text-gray-500 dark:text-gray-400 transition-colors flex items-center justify-center"
          >
            Clear
          </button>
          <button
            onClick={() => handleDigit("0")}
            className="h-12 rounded-xl bg-gray-100 dark:bg-gray-800 hover:bg-gray-200 dark:hover:bg-gray-700 text-lg font-semibold text-gray-800 dark:text-gray-200 transition-colors"
          >
            0
          </button>
          <button
            onClick={onClose}
            className="h-12 rounded-xl bg-red-100 dark:bg-red-900/30 hover:bg-red-200 dark:hover:bg-red-900/50 text-xs font-semibold text-red-600 dark:text-red-400 transition-colors flex items-center justify-center"
          >
            Cancel
          </button>
        </div>
      </div>
    </div>
  );
}
