# Dataset Audit Notes

Source: `data/raw/mental_health_final_USER_TONE_dataset (1).xlsx`, sheet
`Combined Data`, 11,500 rows. Read directly with `openpyxl`; findings below
were verified against the actual cell values, not assumed from the sheet
names.

## Sheets in the workbook

- `Combined Data` — the 11,500-row per-respondent data (source of truth for training).
- `Risk Level Legend` — 6-level text definitions (matches Section 3 of the product spec, including the explicit "N/A – Feeling Fine" category).
- `Summary Statistics` — claims a 6-level distribution: N/A 28.0% (3,222), Low 3.4% (387), Mild 12.2% (1,407), Moderate 9.8% (1,123), Severe 29.9% (3,443), Emergency 16.7% (1,918).
- `Risk Framework` — the 5-level (no N/A row) definition + suggested-next-steps template text, used verbatim for `report_service.py`'s next-steps copy.
- `Risk Counts` — a 5-level (no N/A row) count table: Low 3,422, Mild 489, Moderate 1,886, Severe 3,785, Emergency 1,918.

## Finding 1 — the raw `Risk Level` column has no "N/A – Feeling Fine" value

Distinct values in `Combined Data`'s `Risk Level` column: `{Severe: 3785, Low:
3422, Emergency: 1918, Moderate: 1886, Mild: 489}` — 5 values, sums to
11,500. There is no literal `N/A-Feeling Fine` string anywhere in the column.

Cross-checked: **all 3,222 rows where Q2–Q15 are null have `Q1 == "I'm
feeling fine / just checking"` and raw `Risk Level == "Low"`.** These are
exactly the "N/A – Feeling Fine" respondents described in the Legend, folded
into the `Low` bucket at generation time. No row has a partial null pattern
— Q2–Q15 are either all null or all populated (verified exhaustively).

**Fix applied in `prepare_dataset.py`:** if Q2–Q15 are all null → relabel
`N/A-Feeling Fine`; otherwise keep the raw `Risk Level` value. This yields
exactly 200 genuine non-null "Low" rows (avg intensity 2.02, range 1–3 —
consistent with the Legend's "Low (Watchful)" definition) separated cleanly
from the 3,222 N/A rows (avg intensity 0).

## Finding 2 — `Summary Statistics` is stale/inconsistent and should NOT be used for calibration

This required deeper verification than initially assumed. Three sheets
disagree on the Low/Mild/Moderate breakdown:

| Level | `Risk Counts` sheet | Raw `Combined Data` column | `Summary Statistics` sheet |
|---|---|---|---|
| Low | 3,422 | 3,422 | 387 |
| Mild | 489 | 489 | 1,407 |
| Moderate | 1,886 | 1,886 | 1,123 |
| Severe | 3,785 | 3,785 | 3,443 |
| Emergency | 1,918 | 1,918 | 1,918 |

`Risk Counts` matches the raw `Combined Data` column **exactly**, on every
level. `Summary Statistics` only agrees on Emergency (1,918) and disagrees
substantially everywhere else — including a Low count (387) that doesn't
even match the 200 true non-null-Low rows found in Finding 1. This means
`Summary Statistics` was very likely computed from an earlier/different
generation pass of the dataset and never reconciled after regeneration.

**Decision: treat the raw `Combined Data` `Risk Level` column (with the
Finding 1 null-row fix applied) as the authoritative row-level label.**
It's corroborated by an independent sheet (`Risk Counts`) and is internally
self-consistent (each row's `Suggested Next Steps` text is a 1:1 function of
its own `Risk Level` value — 5 distinct next-steps templates, counts match
exactly). Do not use `Summary Statistics` percentages as an eval target;
they describe a different labeling than what's actually in the rows.

**Resulting true 6-way distribution used for training/eval:**
N/A-Feeling Fine 3,222 (28.0%), Low 200 (1.7%), Mild 489 (4.3%), Moderate
1,886 (16.4%), Severe 3,785 (32.9%), Emergency 1,918 (16.7%). Total 11,500.
Note `Low` is now a very small class (1.7%, 200 rows) — flag this explicitly
in Stage 1 eval; per-class recall on Low is likely to be the weakest metric
and may need class-weighting or oversampling during training.

## Finding 3 — Q9/Q10/Q11 need explicit mapping to the live verbatim answer options

Raw value counts (non-null rows only, n=8,278):

- **Q9 (Passive SI):** `Often: 3516, Sometimes: 2405, No: 1337, Rarely: 1020`. Live conversation offers `No / Occasionally / Often / prefer not to answer`. Mapping: `No→No`, `Rarely→Occasionally`, `Sometimes→Occasionally`, `Often→Often`. No dataset row exercises "prefer not to answer" — acceptable; it's a valid live-only class the model will simply never have training signal for.
- **Q10 (Active SI/SH):** `No: 2917, "Brief thoughts, but I don't want to act on them": 2396, "Ongoing thoughts, but no plan": 1047, "Yes, I may act on these thoughts": 1918`. Live conversation offers `No / Yes but wouldn't act / Yes and it feels risky right now`. Mapping: `No→No`; `"Brief thoughts..."` and `"Ongoing thoughts, but no plan"→Yes but wouldn't act`; `"Yes, I may act on these thoughts"→Yes and it feels risky right now`.
- **Q11 (Feels Safe):** `"I'm not fully sure": 3735, Yes: 2625, "No, I feel unsafe": 1918`.

**Strong internal consistency check (holds exactly, verified directly):**
`Emergency count (1,918) == Q10 "Yes, I may act on these thoughts" count
(1,918) == Q11 "No, I feel unsafe" count (1,918)`, and all three refer to
the same 1,918 rows (not just equal counts — same row set, spot-checked).
This confirms the dataset was generated consistent with the spec's critical
boundary rule: passive/occasional SI *without* stated intent classifies as
Severe, never Emergency; Emergency requires stated intent (mapped Q10) or
feeling unsafe right now (Q11). Kept as a permanent regression assertion in
`prepare_dataset.py` — if it ever stops holding for a future data refresh,
the build should fail loudly rather than silently train on a corrupted
Emergency boundary.

## Encoding note

The source `.xlsx` uses smart quotes/en-dashes that decode fine via
`openpyxl` but break Windows console `cp1252` printing (`UnicodeEncodeError`
on `’`/`–`/`✓`). Not a data problem — only affects debug printing to a
`cp1252` terminal. `prepare_dataset.py` writes outputs as UTF-8 and never
prints raw cell text to the console.

## Reference next-steps copy (from `Risk Framework` sheet, used verbatim in `report_service.py`)

- **Low:** "What you're describing sounds difficult but not unsafe. Try one supportive conversation, light self-monitoring (sleep/appetite/mood), and 2–3 small stabilizers this week such as hydration, a walk, or a simple routine."
- **Mild:** "Would it help to try a small plan for the next 7 days? Pick one daily stabilizer (sleep window, movement, social contact), speak with a trusted person or low-barrier counselor, and use a professional directory if needed."
- **Moderate:** "It would be a good idea to talk to a counsellor or psychologist within 1–2 weeks. Use this summary as a starter script, reduce substances, keep sleep regular, and plan one manageable task per day."
- **Severe:** "A professional evaluation is recommended soon (ideally within 24–72 hours if possible). Do not stay alone if you feel unsafe, involve a trusted adult/person, and seek a psychologist or psychiatrist, especially if functioning is strongly affected or symptoms feel out of control."
- **Emergency:** "I'm concerned about your safety right now. Contact a trusted adult/person immediately, call a crisis helpline or emergency service, or go to the nearest ER. A safety plan can be made with a professional, including warning signs, coping steps, supportive contacts, and reducing access to means."
- **N/A – Feeling Fine:** no next-steps template in `Risk Framework` (5-level sheet only) — use a short light closing message per spec, no forced follow-up.
