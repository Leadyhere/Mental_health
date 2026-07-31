"use client";

import { useEffect, useState } from "react";
import { getReport, getReportPdfUrl } from "@/lib/apiClient";
import { ReportResponse } from "@/lib/types";
import { SummaryCard } from "@/components/report/SummaryCard";

export function ReportClient({ sessionId }: { sessionId: string }) {
  const [report, setReport] = useState<ReportResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    getReport(sessionId)
      .then(setReport)
      .catch(() => setError("Report isn't ready yet, or this session has ended."));
  }, [sessionId]);

  return (
    <div className="min-h-screen px-4 py-10">
      <div className="max-w-2xl mx-auto space-y-6">
        {error && <p className="text-sm text-red-600">{error}</p>}
        {report && (
          <>
            <SummaryCard report={report} />
            <a
              href={getReportPdfUrl(sessionId)}
              className="inline-block rounded-xl bg-blue-600 text-white px-4 py-2 text-sm font-medium"
            >
              Download as PDF
            </a>
          </>
        )}
      </div>
    </div>
  );
}
