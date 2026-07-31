export function NextStepsList({ text }: { text: string }) {
  return (
    <div className="rounded-xl bg-blue-50 dark:bg-blue-950/40 border border-blue-200 dark:border-blue-900 p-4">
      <h3 className="font-semibold text-sm mb-2 text-blue-900 dark:text-blue-100">Suggested Next Steps</h3>
      <p className="text-sm text-blue-950 dark:text-blue-50 whitespace-pre-wrap">{text}</p>
    </div>
  );
}
