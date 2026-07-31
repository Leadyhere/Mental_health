"use client";

import { motion } from "framer-motion";
import { CrisisBannerPayload } from "@/lib/types";

export function CrisisBanner({ banner }: { banner: CrisisBannerPayload }) {
  return (
    <motion.div
      initial={{ y: -40, opacity: 0 }}
      animate={{ y: 0, opacity: 1 }}
      transition={{ duration: 0.25 }}
      className="fixed top-0 left-0 right-0 z-50 bg-red-700 text-white px-4 py-4 shadow-lg"
    >
      <div className="max-w-2xl mx-auto">
        <p className="font-semibold mb-2">{banner.message}</p>
        <ul className="space-y-1 text-sm">
          {banner.resources.map((r) => (
            <li key={r.name}>
              <span className="font-semibold">{r.name}:</span> {r.phone} - {r.description}
            </li>
          ))}
        </ul>
      </div>
    </motion.div>
  );
}
