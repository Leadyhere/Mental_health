export function PromptChips({
  options,
  onSelect,
  disabled,
}: {
  options: string[];
  onSelect: (value: string) => void;
  disabled?: boolean;
}) {
  return (
    <div className="flex flex-wrap gap-2">
      {options.map((option) => (
        <button
          key={option}
          disabled={disabled}
          onClick={() => onSelect(option)}
          className="rounded-full border border-blue-500 text-blue-600 dark:text-blue-400 px-3 py-1.5 text-sm hover:bg-blue-50 dark:hover:bg-blue-950 disabled:opacity-40"
        >
          {option}
        </button>
      ))}
    </div>
  );
}
