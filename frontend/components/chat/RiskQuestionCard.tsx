import { PromptChips } from "./PromptChips";

/**
 * Renders the two verbatim risk-sensing questions (passive/active SI).
 * Kept as its own component -- separate from the generic chip renderer --
 * so these prompts are visually and structurally distinct from anything
 * that could be paraphrased or templated.
 */
export function RiskQuestionCard({
  prompt,
  options,
  onSelect,
  disabled,
}: {
  prompt: string;
  options: string[];
  onSelect: (value: string) => void;
  disabled?: boolean;
}) {
  return (
    <div className="rounded-xl border border-amber-300 dark:border-amber-700 bg-amber-50 dark:bg-amber-950/40 p-4 space-y-3">
      <p className="text-sm text-amber-900 dark:text-amber-100">{prompt}</p>
      <PromptChips options={options} onSelect={onSelect} disabled={disabled} />
    </div>
  );
}
