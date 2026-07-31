const COLORS: Record<string, string> = {
  "N/A-Feeling Fine": "bg-gray-200 text-gray-800",
  Low: "bg-green-200 text-green-900",
  Mild: "bg-yellow-200 text-yellow-900",
  Moderate: "bg-orange-200 text-orange-900",
  Severe: "bg-red-200 text-red-900",
  Emergency: "bg-red-700 text-white",
};

export function RiskLevelBadge({ riskLevel }: { riskLevel: string }) {
  const cls = COLORS[riskLevel] ?? "bg-gray-200 text-gray-800";
  return <span className={`inline-block rounded-full px-3 py-1 text-sm font-semibold ${cls}`}>{riskLevel}</span>;
}
