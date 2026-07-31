import { ReportResponse } from "@/lib/types";
import { NextStepsList } from "./NextStepsList";
import { RiskLevelBadge } from "./RiskLevelBadge";

export function SummaryCard({ report }: { report: ReportResponse }) {
  return (
    <div className="space-y-4">
      <div className="flex items-center gap-3">
        <h2 className="text-lg font-semibold">Session Summary</h2>
        {report.risk_level && <RiskLevelBadge riskLevel={report.risk_level} />}
      </div>

      {report.rationale && (
        <p className="text-sm text-gray-600 dark:text-gray-400 italic">{report.rationale}</p>
      )}

      {report.narrative_highlights && (
        <div>
          <h3 className="font-semibold text-sm mb-1">Narrative Highlights</h3>
          <ul className="text-sm list-disc list-inside space-y-0.5">
            {Object.entries(report.narrative_highlights).map(([key, value]) => (
              <li key={key}>
                <span className="font-medium">{key}:</span>{" "}
                {Array.isArray(value) ? value.join(", ") : String(value ?? "")}
              </li>
            ))}
          </ul>
        </div>
      )}

      {report.structured_answers.length > 0 && (
        <div>
          <h3 className="font-semibold text-sm mb-1">Structured Answers</h3>
          <ul className="text-sm space-y-1">
            {report.structured_answers.map((qa, i) => (
              <li key={i}>
                <span className="text-gray-500">{qa.question}</span> - <span className="font-medium">{qa.answer}</span>
              </li>
            ))}
          </ul>
        </div>
      )}

      {report.suggested_next_steps && <NextStepsList text={report.suggested_next_steps} />}
    </div>
  );
}
