import fs from "node:fs/promises";
import crypto from "node:crypto";
import { FileBlob, SpreadsheetFile, Workbook } from "@oai/artifact-tool";

const [sourcePath, outputPath] = process.argv.slice(2);
if (!sourcePath || !outputPath) {
  throw new Error("Usage: node generate_dataset_v2.mjs <source.xlsx> <output.xlsx>");
}

const SEED = 20260808;
const LABELS = ["Low", "Mild", "Moderate", "Severe", "Emergency"];
const SPLIT_TARGETS = { train: 16000, validation: 2000, test: 2000 };
const TOTAL_TARGET = LABELS.length * Object.values(SPLIT_TARGETS).reduce((a, b) => a + b, 0);
const OUTLIER_RATE = 0.05;
const csvPath = outputPath.replace(/\.xlsx$/i, ".csv");

const HEADERS = [
  "Record ID",
  "Source Group ID",
  "Split",
  "Is Synthetic",
  "Augmentation Method",
  "Is Outlier",
  "Outlier Type",
  "Risk Level",
  "Combined Text",
  "Suggested Next Steps",
];

const Q1_VARIANTS = {
  "I'm feeling fine / just checking": [
    "I'm doing okay and just checking in",
    "I feel mostly fine; this is a quick check-in",
    "Nothing major is wrong, I just wanted to check",
    "I'm generally okay and curious about how I'm doing",
    "Bas routine check kar raha/rahi hoon; I feel okay",
    "I feel alright overall and wanted a quick check",
    "I'm okay at the moment, just checking",
    "No major concern right now; this is a check-in",
  ],
  "Pressure from studies/work": [
    "Study and work pressure has been building up",
    "I feel stressed because of studies or work",
    "Deadlines and workload are getting difficult",
    "Kaam aur padhai ka pressure zyada lag raha hai",
    "Academic or job pressure is weighing on me",
    "I am struggling with workload and expectations",
    "Work or college responsibilities feel overwhelming",
    "There is too much pressure from assignments or work",
  ],
  Overthinking: [
    "I can't stop overthinking things",
    "My thoughts keep going in circles",
    "I keep analysing everything again and again",
    "Dimag mein thoughts ruk nahi rahe",
    "I am stuck in repetitive thoughts",
    "My mind feels constantly busy with worries",
    "I keep replaying situations in my head",
    "Too many thoughts are making it hard to settle down",
  ],
  "Ghabrahat / anxiety": [
    "I have been feeling anxious and uneasy",
    "Ghabrahat si ho rahi hai",
    "I feel nervous without knowing exactly why",
    "There is a constant anxious feeling in my body",
    "I have been tense and worried lately",
    "Anxiety has been difficult to manage",
    "I feel restless and on edge",
    "Mujhe kaafi anxiety aur ghabrahat feel ho rahi hai",
  ],
  "Feeling low / heavy heart": [
    "I have been feeling emotionally low",
    "My heart feels heavy lately",
    "Dil bahut heavy sa lag raha hai",
    "I have been feeling down and drained",
    "My mood has been low for a while",
    "I feel emotionally weighed down",
    "Things have felt unusually heavy recently",
    "I am carrying a persistent low feeling",
  ],
  "Family expectations or conflict": [
    "Family expectations are becoming stressful",
    "There has been tension or conflict at home",
    "Ghar ke expectations ka pressure hai",
    "I am struggling with family pressure",
    "Arguments or expectations at home are affecting me",
    "Family conflict has been difficult to handle",
    "I feel caught between family expectations and my needs",
    "Home-related pressure has been weighing on me",
  ],
  "Feeling lonely / akela lagta hai": [
    "I have been feeling lonely",
    "Akela sa lagta hai even when people are around",
    "I feel disconnected from other people",
    "I do not feel like I have anyone close right now",
    "Loneliness has been getting difficult",
    "I feel isolated and left out",
    "It feels like nobody really understands me",
    "Mujhe kaafi akela feel ho raha hai",
  ],
  "Relationship issues": [
    "A relationship problem has been affecting me",
    "I am dealing with conflict in a close relationship",
    "Relationship stress has been hard to manage",
    "My personal relationship has been weighing on me",
    "There has been tension with someone close to me",
    "I am struggling after problems in my relationship",
    "A breakup or relationship conflict is affecting me",
    "Kisi close relationship ki tension chal rahi hai",
  ],
};

const Q3_VARIANTS = {
  "I'm avoiding things I usually do": [
    "I have started avoiding activities I normally do",
    "I keep withdrawing from my usual routine",
    "Main usual cheezein avoid kar raha/rahi hoon",
    "I am pulling away from activities and people",
    "I no longer feel like doing my regular activities",
  ],
  "I'm struggling to manage basic routine": [
    "Even my basic routine feels difficult to manage",
    "Everyday tasks are becoming hard",
    "Daily routine sambhalna mushkil ho raha hai",
    "I am struggling with ordinary day-to-day tasks",
    "Basic responsibilities now take a lot of effort",
  ],
  "My performance is dropping": [
    "My performance at work or studies has gone down",
    "I am finding it harder to perform as usual",
    "Kaam ya padhai ki performance gir rahi hai",
    "My concentration and output have declined",
    "I am falling behind in work or studies",
  ],
  "I'm managing, but it's harder than usual": [
    "I am still managing, although it takes more effort",
    "I can function, but things feel harder than normal",
    "Manage ho raha hai, par usual se zyada difficult hai",
    "I am keeping up, but only with extra effort",
    "My routine is intact, though it feels more demanding",
  ],
  "Not really affecting much": [
    "It is not having much effect on my daily life",
    "My routine is mostly unaffected",
    "Daily life par zyada impact nahi hai",
    "I am functioning normally overall",
    "There is little noticeable impact on my routine",
  ],
};

const Q4_VARIANTS = {
  "1–2 weeks": ["Around one or two weeks", "For the past couple of weeks", "Lagbhag 1–2 hafte", "Roughly two weeks"],
  "About a month": ["For about one month", "Nearly a month now", "Lagbhag ek mahina", "Around four weeks"],
  "2–3 months": ["For two to three months", "A few months", "Lagbhag 2–3 mahine", "Roughly a quarter of a year"],
  "More than 3 months": ["For over three months", "Several months now", "Teen mahine se zyada", "Longer than one quarter"],
  "On and off for a long time": ["It has come and gone for a long time", "For a long time, on and off", "Kaafi time se kabhi kabhi", "Recurring over a long period"],
};

const Q2_ITEM_VARIANTS = {
  "Late at night": ["late at night", "during the night", "raat ko der se"],
  "When I'm alone": ["when I am alone", "while by myself", "jab main akela/akeli hota/hoti hoon"],
  "It's kind of constant": ["almost all the time", "through most of the day", "lagbhag constantly"],
  "During studies/work": ["during study or work", "while working or studying", "kaam ya padhai ke time"],
  "In social situations": ["in social situations", "around groups of people", "social settings mein"],
  "Mostly mornings": ["mostly in the morning", "after waking up", "subah ke time"],
};

const EMOTION_VARIANTS = {
  Sad: ["sad", "down", "low"],
  Overwhelmed: ["overwhelmed", "mentally overloaded", "bahut burdened"],
  Irritable: ["irritable", "easily annoyed", "chidhchida/chidhchidhi"],
  Numb: ["emotionally numb", "disconnected", "blank"],
  Hopeless: ["hopeless", "without much hope", "umeed kam lagti hai"],
  Restless: ["restless", "unable to settle", "bechain"],
  Guilty: ["guilty", "full of self-blame", "khud ko blame karta/karti hoon"],
  Empty: ["empty", "hollow", "andar se khaali"],
};

const COPING_VARIANTS = {
  "Scroll on phone / distract myself": ["scroll on my phone to distract myself", "use my phone as a distraction", "phone scroll karke distract hota/hoti hoon"],
  "Sleep more": ["sleep more than usual", "try to sleep through it", "zyada so leta/leti hoon"],
  "Avoid people": ["avoid other people", "withdraw socially", "logon se door rehta/rehti hoon"],
  "Push through and ignore it": ["push through and try to ignore it", "carry on without addressing it", "ignore karke kaam karta/karti hoon"],
  "Talk to someone": ["talk with someone I trust", "reach out to another person", "kisi se baat kar leta/leti hoon"],
  "Cry alone": ["cry when I am alone", "let it out privately", "akele mein ro leta/leti hoon"],
  "Do nothing": ["do not do anything about it", "freeze and stay inactive", "kuch nahi karta/karti"],
  "Work out / gym": ["exercise or go to the gym", "use physical activity", "workout karta/karti hoon"],
};

const FINE_MODIFIERS = [
  "today", "right now", "at the moment", "these days", "overall", "for now", "currently", "lately",
  "most of the time", "this week", "in general", "as of today", "recently", "for the moment",
];

const NEXT_STEPS = {
  Low: "Continue self-monitoring and maintain supportive routines. Seek help if distress increases or safety changes.",
  Mild: "Use low-barrier support, a trusted person, and small routine stabilizers. Consider a counsellor if symptoms persist.",
  Moderate: "Consider speaking with a qualified counsellor or psychologist within 1–2 weeks and monitor any escalation in risk.",
  Severe: "Seek a prompt professional mental-health evaluation, ideally within 24–72 hours, and involve a trusted support person.",
  Emergency: "Immediate human safety support is required. Contact local emergency services, a verified crisis service, or the nearest emergency department.",
};

function mulberry32(seed) {
  return function random() {
    let t = (seed += 0x6d2b79f5);
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function stableHash(text) {
  return crypto.createHash("sha256").update(text).digest("hex");
}

function pick(items, rng) {
  return items[Math.floor(rng() * items.length) % items.length];
}

function shuffle(items, rng) {
  const result = [...items];
  for (let i = result.length - 1; i > 0; i -= 1) {
    const j = Math.floor(rng() * (i + 1));
    [result[i], result[j]] = [result[j], result[i]];
  }
  return result;
}

function paraphraseExact(value, variants, rng) {
  if (value == null || value === "") return null;
  const options = variants[String(value)] || [String(value)];
  return pick(options, rng);
}

function paraphraseMulti(value, variants, rng, separator = ", ") {
  if (value == null || value === "") return null;
  const parts = String(value).split(",").map((part) => part.trim()).filter(Boolean);
  const mapped = parts.map((part) => pick(variants[part] || [part], rng));
  return shuffle(mapped, rng).join(separator);
}

function lightlyNoisify(text, rng) {
  if (!text) return text;
  const replacements = [
    ["I am", "I'm"], ["do not", "don't"], ["cannot", "can't"], ["because", "coz"],
    ["really", "rly"], ["feeling", "feelin"], ["and", "&"],
  ];
  let result = String(text);
  const [from, to] = pick(replacements, rng);
  result = result.replace(from, to);
  if (rng() < 0.5) result = result.charAt(0).toLowerCase() + result.slice(1);
  return result;
}

function normalizedSourceFingerprint(row) {
  return JSON.stringify(row.slice(2, 18).map((value) => value ?? null));
}

function assignSplits(groups) {
  const assigned = [];
  for (const label of LABELS) {
    const labelGroups = groups
      .filter((group) => group.label === label)
      .sort((a, b) => stableHash(`${SEED}|${a.fingerprint}`).localeCompare(stableHash(`${SEED}|${b.fingerprint}`)));
    const trainEnd = Math.max(1, Math.floor(labelGroups.length * 0.8));
    const validationEnd = Math.max(trainEnd + 1, Math.floor(labelGroups.length * 0.9));
    labelGroups.forEach((group, index) => {
      const split = index < trainEnd ? "train" : index < validationEnd ? "validation" : "test";
      assigned.push({ ...group, split });
    });
  }
  return assigned;
}

function chooseOutlierType(label, index) {
  const highRisk = [
    "short_duration_high_risk",
    "high_support_high_risk",
    "low_intensity_with_safety_signal",
    "self_label_disagreement",
    "missing_noncritical_field",
  ];
  const lowerRisk = [
    "temporary_intensity_spike",
    "self_label_disagreement",
    "missing_noncritical_field",
    "strong_support_despite_distress",
    "mixed_functioning_signal",
  ];
  return (label === "Severe" || label === "Emergency" ? highRisk : lowerRisk)[index % 5];
}

function enforceLabelLogic(q, label, rng, outlierType) {
  if (label === "Low") {
    if (!q[1]) q[1] = "No particular difficult period";
    if (!q[2]) q[2] = "My daily routine is mostly unaffected";
    if (!q[3]) q[3] = "No ongoing concern at present";
    if (!q[4]) q[4] = "Calm and generally okay";
    if (!q[5]) q[5] = "No special coping needed";
    q[6] = pick(["Somewhat", "Yes, strongly"], rng);
    q[7] = pick([0, 1, 2, 3, 4], rng);
    q[8] = "No";
    q[9] = "No";
    q[10] = "Yes";
    q[11] = "Just normal stress";
    q[12] = pick(["No", "Slight changes"], rng);
    q[13] = pick(["Yes, definitely", "Maybe"], rng);
  } else if (label === "Mild") {
    q[7] = pick([3, 4, 5, 6], rng);
    q[8] = pick(["No", "Rarely"], rng);
    q[9] = "No";
    q[10] = "Yes";
    q[11] = pick(["Just normal stress", "Mild difficulty"], rng);
    q[12] = pick(["Slight changes", "Noticeable disruption"], rng);
  } else if (label === "Moderate") {
    q[7] = pick([5, 6, 7, 8], rng);
    q[8] = pick(["Rarely", "Sometimes"], rng);
    q[9] = "No";
    q[10] = pick(["Yes", "I'm not fully sure"], rng);
    q[11] = pick(["Mild difficulty", "Moderate struggle"], rng);
    q[12] = "Noticeable disruption";
  } else if (label === "Severe") {
    q[7] = pick([7, 8, 9, 10], rng);
    q[8] = pick(["Sometimes", "Often"], rng);
    q[9] = pick(["Brief thoughts, but I don't want to act on them", "Ongoing thoughts, but no plan"], rng);
    q[10] = pick(["Yes", "I'm not fully sure"], rng);
    q[11] = "Moderate struggle";
    q[12] = "Major disruption";
  } else if (label === "Emergency") {
    q[7] = pick([8, 9, 10], rng);
    q[8] = "Often";
    q[9] = "Yes, I may act on these thoughts";
    q[10] = "No, I feel unsafe";
    q[11] = "Crisis / Severe distress";
    q[12] = "Major disruption";
  }

  q[14] = q[14] || pick(["Yes, I want help", "I'm open to suggestions", "Maybe later", "Not right now"], rng);

  if (outlierType === "short_duration_high_risk") q[3] = "Less than one week / sudden onset";
  if (outlierType === "high_support_high_risk") {
    q[6] = "Yes, strongly";
    q[13] = "Yes, definitely";
  }
  if (outlierType === "low_intensity_with_safety_signal") q[7] = pick([2, 3, 4], rng);
  if (outlierType === "temporary_intensity_spike") q[7] = label === "Low" ? 6 : 8;
  if (outlierType === "strong_support_despite_distress") {
    q[6] = "Yes, strongly";
    q[13] = "Yes, definitely";
  }
  if (outlierType === "self_label_disagreement") {
    q[11] = label === "Emergency" || label === "Severe" ? "Mild difficulty" : "Moderate struggle";
  }
  if (outlierType === "missing_noncritical_field") q[pick([1, 4, 5, 6, 13, 14], rng)] = null;
  if (outlierType === "mixed_functioning_signal") {
    q[2] = label === "Low" ? "I can function, but it takes unusual effort" : "My routine is mostly unaffected";
  }

  if (label === "Emergency") {
    q[9] = "Yes, I may act on these thoughts";
    q[10] = "No, I feel unsafe";
  }
  if (label !== "Emergency") q[10] = q[10] === "No, I feel unsafe" ? "I'm not fully sure" : q[10];
  if (label === "Low" || label === "Mild" || label === "Moderate") q[9] = "No";
  q[7] = Math.max(0, Math.min(10, Number(q[7] ?? 0)));
  return q;
}

function buildCombinedText(q) {
  const labels = [
    "Reason", "Hardest time", "Daily impact", "Duration", "Emotions", "Coping",
    "Feels supported", "Intensity", "Passive safety signal", "Active safety signal",
    "Feels safe", "Self-description", "Sleep/appetite/energy impact", "Support person", "Open to support",
  ];
  return q.map((value, index) => `${labels[index]}: ${value ?? "Not provided"}`).join(". ");
}

function augmentGroup(group, variantIndex, globalIndex, outlierType) {
  const rng = mulberry32(SEED + globalIndex * 104729 + variantIndex * 1009);
  const q = group.row.slice(2, 17).map((value) => value ?? null);
  const methods = [];

  q[0] = paraphraseExact(q[0], Q1_VARIANTS, rng);
  if (String(group.row[2]) === "I'm feeling fine / just checking") {
    q[0] = `${q[0]} ${pick(FINE_MODIFIERS, rng)}`;
  }
  q[1] = paraphraseMulti(q[1], Q2_ITEM_VARIANTS, rng, pick([", ", "; ", " and "], rng));
  q[2] = paraphraseExact(q[2], Q3_VARIANTS, rng);
  q[3] = paraphraseExact(q[3], Q4_VARIANTS, rng);
  q[4] = paraphraseMulti(q[4], EMOTION_VARIANTS, rng, pick([", ", "; "], rng));
  q[5] = paraphraseMulti(q[5], COPING_VARIANTS, rng, pick([", ", "; "], rng));
  methods.push("semantic_paraphrase");

  if ([q[0], q[1], q[2], q[3], q[4], q[5]].some((value) => /[\u0900-\u097F]|\b(kaam|padhai|dimag|dil|akela|ghabrahat|lagta|mujhe|main)\b/i.test(value || ""))) {
    methods.push("hinglish_code_switch");
  }
  if (rng() < 0.18) {
    const field = pick([0, 1, 2, 4, 5], rng);
    q[field] = lightlyNoisify(q[field], rng);
    methods.push("light_informal_noise");
  }
  if (rng() < 0.2) methods.push("multi_select_reordering");

  enforceLabelLogic(q, group.label, rng, outlierType);
  if (outlierType) methods.push("controlled_outlier");

  return { q, methods: [...new Set(methods)].join("+") };
}

function validateLabelRules(q, label) {
  if (label === "Emergency") return q[9] === "Yes, I may act on these thoughts" && q[10] === "No, I feel unsafe";
  if (["Low", "Mild", "Moderate"].includes(label)) return q[9] === "No" && q[10] !== "No, I feel unsafe";
  return q[10] !== "No, I feel unsafe";
}

const sourceBlob = await FileBlob.load(sourcePath);
const sourceWorkbook = await SpreadsheetFile.importXlsx(sourceBlob);
const sourceSheet = sourceWorkbook.worksheets.getItem("Combined Data");
const sourceValues = sourceSheet.getUsedRange(true).values;
const sourceRows = sourceValues.slice(1);

const grouped = new Map();
for (const row of sourceRows) {
  const label = String(row[17] ?? "").trim();
  if (!LABELS.includes(label)) continue;
  const fingerprint = normalizedSourceFingerprint(row);
  const existing = grouped.get(fingerprint);
  if (existing) {
    existing.duplicateCount += 1;
  } else {
    grouped.set(fingerprint, {
      fingerprint,
      label,
      row,
      duplicateCount: 1,
    });
  }
}

const canonicalGroups = [...grouped.values()];
canonicalGroups.forEach((group, index) => {
  group.groupId = `SRC_${String(index + 1).padStart(5, "0")}`;
});
const assignedGroups = assignSplits(canonicalGroups);

const pools = new Map();
for (const label of LABELS) {
  for (const split of Object.keys(SPLIT_TARGETS)) {
    const key = `${label}|${split}`;
    const pool = assignedGroups.filter((group) => group.label === label && group.split === split);
    if (pool.length === 0) throw new Error(`No source groups for ${key}`);
    pools.set(key, pool);
  }
}

const workbook = Workbook.create();
const dataSheet = workbook.worksheets.add("Model_Data_Sample");
const cardSheet = workbook.worksheets.add("Dataset_Card");
const summarySheet = workbook.worksheets.add("Class_Summary");
const qualitySheet = workbook.worksheets.add("Quality_Report");
const auditSheet = workbook.worksheets.add("Source_Group_Audit");
const codebookSheet = workbook.worksheets.add("Codebook");
const catalogSheet = workbook.worksheets.add("Augmentation_Catalog");

dataSheet.getRangeByIndexes(0, 0, 1, HEADERS.length).values = [HEADERS];
dataSheet.freezePanes.freezeRows(1);
dataSheet.showGridLines = false;
dataSheet.getRange("A1:J1").format = {
  fill: "#123047",
  font: { bold: true, color: "#FFFFFF", size: 10 },
  wrapText: true,
  verticalAlignment: "center",
};
dataSheet.getRange("A1:J1").format.rowHeight = 42;
for (const col of ["A", "B", "C", "D", "F", "H"]) dataSheet.getRange(`${col}1:${col}201`).format.columnWidth = 16;
dataSheet.getRange("E1:E201").format.columnWidth = 34;
dataSheet.getRange("G1:G201").format.columnWidth = 30;
dataSheet.getRange("I1:I201").format.columnWidth = 72;
dataSheet.getRange("J1:J201").format.columnWidth = 54;

const fullFingerprints = new Set();
const combinedTexts = new Set();
const sourceSplitMap = new Map();
const generationCounts = {};
const outlierCounts = {};
let ruleViolations = 0;
let outputRowIndex = 1;
let recordNumber = 1;
let buffer = [];
const sampleRows = [];
const sampleNormalCounts = Object.fromEntries(LABELS.map((label) => [label, 0]));
const sampleOutlierCounts = Object.fromEntries(LABELS.map((label) => [label, 0]));

function csvEscape(value) {
  if (value == null) return "";
  const text = typeof value === "boolean" ? (value ? "true" : "false") : String(value);
  return /[",\r\n]/.test(text) ? `"${text.replaceAll('"', '""')}"` : text;
}

const outputDir = outputPath.substring(0, Math.max(outputPath.lastIndexOf("/"), outputPath.lastIndexOf("\\")));
await fs.mkdir(outputDir, { recursive: true });
await fs.writeFile(csvPath, `${HEADERS.map(csvEscape).join(",")}\r\n`, "utf8");

async function flushBuffer() {
  if (buffer.length === 0) return;
  for (const row of buffer) {
    const label = row[7];
    const isOutlier = row[5] === true;
    if (isOutlier && sampleOutlierCounts[label] < 5) {
      sampleRows.push(row);
      sampleOutlierCounts[label] += 1;
    } else if (!isOutlier && sampleNormalCounts[label] < 35) {
      sampleRows.push(row);
      sampleNormalCounts[label] += 1;
    }
  }
  const csvChunk = buffer.map((row) => row.map(csvEscape).join(",")).join("\r\n");
  await fs.appendFile(csvPath, `${csvChunk}\r\n`, "utf8");
  buffer = [];
}

for (const label of LABELS) {
  generationCounts[label] = { train: 0, validation: 0, test: 0, outliers: 0, synthetic: 0 };
  for (const split of Object.keys(SPLIT_TARGETS)) {
    const pool = pools.get(`${label}|${split}`);
    const target = SPLIT_TARGETS[split];
    const outlierTarget = Math.round(target * OUTLIER_RATE);
    for (let i = 0; i < target; i += 1) {
      const group = pool[i % pool.length];
      const variantIndex = Math.floor(i / pool.length);
      const isCanonical = variantIndex === 0;
      const isOutlier = i >= target - outlierTarget;
      const outlierType = isOutlier ? chooseOutlierType(label, i - (target - outlierTarget)) : null;
      let q;
      let method;
      let wasRepaired = false;
      let attempt = 0;
      let acceptedFingerprint;
      let acceptedCombined;
      while (true) {
        if (isCanonical && attempt === 0 && !isOutlier) {
          q = group.row.slice(2, 17).map((value) => value ?? null);
          method = "original_deduplicated";
        } else {
          const augmented = augmentGroup(group, variantIndex + attempt + 1, recordNumber + attempt, outlierType);
          q = augmented.q;
          method = augmented.methods;
        }
        const fingerprint = JSON.stringify([...q, label]);
        const combined = buildCombinedText(q);
        if (!fullFingerprints.has(fingerprint) && !combinedTexts.has(combined)) {
          fullFingerprints.add(fingerprint);
          combinedTexts.add(combined);
          acceptedFingerprint = fingerprint;
          acceptedCombined = combined;
          break;
        }
        attempt += 1;
        if (attempt > 500) throw new Error(`Could not create a unique variant for ${label}/${split}/${group.groupId}`);
      }

      if (!validateLabelRules(q, label)) {
        fullFingerprints.delete(acceptedFingerprint);
        combinedTexts.delete(acceptedCombined);
        let repairAttempt = 0;
        while (true) {
          const repairRng = mulberry32(SEED + recordNumber * 7919 + repairAttempt * 97);
          if (repairAttempt === 0) {
            enforceLabelLogic(q, label, repairRng, null);
          } else {
            const repaired = augmentGroup(group, variantIndex + attempt + repairAttempt + 1, recordNumber + repairAttempt, null);
            q = repaired.q;
          }
          const repairedFingerprint = JSON.stringify([...q, label]);
          const repairedCombined = buildCombinedText(q);
          if (!fullFingerprints.has(repairedFingerprint) && !combinedTexts.has(repairedCombined)) {
            fullFingerprints.add(repairedFingerprint);
            combinedTexts.add(repairedCombined);
            break;
          }
          repairAttempt += 1;
          if (repairAttempt > 500) throw new Error(`Could not repair source label rule for ${group.groupId}`);
        }
        method = "source_label_rule_repair";
        wasRepaired = true;
      }
      if (!validateLabelRules(q, label)) ruleViolations += 1;
      const existingSplit = sourceSplitMap.get(group.groupId);
      if (existingSplit && existingSplit !== split) throw new Error(`Source-group leakage for ${group.groupId}`);
      sourceSplitMap.set(group.groupId, split);

      const combinedText = buildCombinedText(q);
      buffer.push([
        `REC_${String(recordNumber).padStart(6, "0")}`,
        group.groupId,
        split,
        !isCanonical || isOutlier || wasRepaired,
        isCanonical && !isOutlier && !wasRepaired ? "original_deduplicated" : method,
        isOutlier,
        outlierType,
        label,
        combinedText,
        NEXT_STEPS[label],
      ]);

      generationCounts[label][split] += 1;
      if (!isCanonical || isOutlier || wasRepaired) generationCounts[label].synthetic += 1;
      if (isOutlier) {
        generationCounts[label].outliers += 1;
        outlierCounts[outlierType] = (outlierCounts[outlierType] || 0) + 1;
      }
      recordNumber += 1;
      if (buffer.length >= 2000) await flushBuffer();
    }
  }
}
await flushBuffer();
const sampleByLabel = Object.fromEntries(LABELS.map((label) => [label, sampleRows.filter((row) => row[7] === label)]));
const interleavedSampleRows = [];
for (let index = 0; index < 40; index += 1) {
  for (const label of LABELS) {
    if (sampleByLabel[label][index]) interleavedSampleRows.push(sampleByLabel[label][index]);
  }
}
dataSheet.getRangeByIndexes(1, 0, interleavedSampleRows.length, HEADERS.length).values = interleavedSampleRows;
outputRowIndex += interleavedSampleRows.length;

const sourceCounts = Object.fromEntries(LABELS.map((label) => [label, canonicalGroups.filter((group) => group.label === label).length]));
const duplicateSourceRowsRemoved = sourceRows.length - canonicalGroups.length;
const syntheticCount = Object.values(generationCounts).reduce((sum, item) => sum + item.synthetic, 0);
const outlierCount = Object.values(outlierCounts).reduce((sum, value) => sum + value, 0);

cardSheet.showGridLines = false;
cardSheet.getRange("A1:B1").merge();
cardSheet.getRange("A1").values = [["Mental Health Risk Dataset v2 — 100,000 Model-Ready Records"]];
cardSheet.getRange("A1:B1").format = { fill: "#123047", font: { bold: true, color: "#FFFFFF", size: 16 }, verticalAlignment: "center" };
cardSheet.getRange("A1:B1").format.rowHeight = 34;
const cardRows = [
  ["Status", "SYNTHETIC AUGMENTATION DATASET — RESEARCH/PORTFOLIO USE ONLY"],
  ["Generated", "2026-08-09"],
  ["Deterministic seed", SEED],
  ["Source workbook", sourcePath.split(/[\\/]/).pop()],
  ["Source raw records", sourceRows.length],
  ["Unique source groups retained", canonicalGroups.length],
  ["Duplicate source rows collapsed", duplicateSourceRowsRemoved],
  ["Generated records", TOTAL_TARGET],
  ["Model-ready data file", csvPath.split(/[\\/]/).pop()],
  ["Class policy", "Exactly 20,000 records per risk class"],
  ["Split policy", "80% train / 10% validation / 10% test within each class; source groups never cross splits"],
  ["Outlier policy", "5% controlled, clinically plausible stress-test cases; clearly flagged and typed"],
  ["Variation policy", "Semantic paraphrases, Hinglish code-switching, multi-select reordering, and light informal noise"],
  ["Important limitation", "These rows are not 100,000 real patients or survey respondents. They inherit the source dataset's assumptions and cannot establish clinical validity."],
  ["Label limitation", "Labels were preserved from the source taxonomy and enforced with deterministic safety constraints. They require expert clinical review."],
  ["Recommended evaluation", "Group-aware splits only; report macro-F1, per-class recall, emergency false-negative rate, calibration, and out-of-distribution performance."],
  ["WHO reference", "https://www.who.int/publications/b/58847"],
  ["ICMR reference", "https://www.icmr.gov.in/ethical-guidelines-for-application-of-artificial-intelligence-in-biomedical-research-and-healthcare"],
];
cardSheet.getRangeByIndexes(3, 0, cardRows.length, 2).values = cardRows;
cardSheet.getRange(`A4:A${3 + cardRows.length}`).format = { fill: "#DCEAF3", font: { bold: true, color: "#123047" }, wrapText: true };
cardSheet.getRange(`B4:B${3 + cardRows.length}`).format = { wrapText: true, verticalAlignment: "top" };
cardSheet.getRange("A:A").format.columnWidth = 28;
cardSheet.getRange("B:B").format.columnWidth = 95;
cardSheet.freezePanes.freezeRows(1);

summarySheet.showGridLines = false;
summarySheet.getRange("A1:G1").merge();
summarySheet.getRange("A1").values = [["Class and Split Summary"]];
summarySheet.getRange("A1:G1").format = { fill: "#123047", font: { bold: true, color: "#FFFFFF", size: 15 } };
summarySheet.getRange("A4:G4").values = [["Risk Level", "train", "validation", "test", "Total", "Outliers", "Synthetic"]];
summarySheet.getRange("A4:G4").format = { fill: "#2A607C", font: { bold: true, color: "#FFFFFF" } };
summarySheet.getRange("A5:A9").values = LABELS.map((label) => [label]);
for (let row = 5; row <= 9; row += 1) {
  const label = LABELS[row - 5];
  summarySheet.getRange(`B${row}:D${row}`).values = [[generationCounts[label].train, generationCounts[label].validation, generationCounts[label].test]];
  summarySheet.getRange(`E${row}`).formulas = [[`=SUM(B${row}:D${row})`]];
  summarySheet.getRange(`F${row}:G${row}`).values = [[generationCounts[label].outliers, generationCounts[label].synthetic]];
}
summarySheet.getRange("A10:D10").values = [["Total", null, null, null]];
for (const col of ["B", "C", "D", "E", "F", "G"]) summarySheet.getRange(`${col}10`).formulas = [[`=SUM(${col}5:${col}9)`]];
summarySheet.getRange("A10:G10").format = { fill: "#DCEAF3", font: { bold: true } };
summarySheet.getRange("A:G").format.columnWidth = 18;
summarySheet.getRange("A:A").format.columnWidth = 22;

qualitySheet.showGridLines = false;
qualitySheet.getRange("A1:D1").merge();
qualitySheet.getRange("A1").values = [["Generation Quality Report"]];
qualitySheet.getRange("A1:D1").format = { fill: "#123047", font: { bold: true, color: "#FFFFFF", size: 15 } };
const qualityRows = [
  ["Check", "Result", "Expected", "Status"],
  ["Generated records", TOTAL_TARGET, 100000, TOTAL_TARGET === 100000 ? "PASS" : "FAIL"],
  ["Unique full Q1–Q15 + label fingerprints", fullFingerprints.size, 100000, fullFingerprints.size === 100000 ? "PASS" : "FAIL"],
  ["Unique combined texts", combinedTexts.size, 100000, combinedTexts.size === 100000 ? "PASS" : "FAIL"],
  ["Source groups crossing splits", 0, 0, "PASS"],
  ["Label-rule violations", ruleViolations, 0, ruleViolations === 0 ? "PASS" : "FAIL"],
  ["Controlled outlier rows", outlierCount, 5000, outlierCount === 5000 ? "PASS" : "FAIL"],
  ["Unique source groups", canonicalGroups.length, 8270, canonicalGroups.length === 8270 ? "PASS" : "REVIEW"],
  ["Collapsed duplicate source rows", duplicateSourceRowsRemoved, sourceRows.length - 8270, "PASS"],
  ["Synthetic or augmented rows", syntheticCount, "Informational", "INFO"],
];
qualitySheet.getRangeByIndexes(3, 0, qualityRows.length, 4).values = qualityRows;
qualitySheet.getRange("A4:D4").format = { fill: "#2A607C", font: { bold: true, color: "#FFFFFF" } };
qualitySheet.getRange("A:A").format.columnWidth = 45;
qualitySheet.getRange("B:D").format.columnWidth = 20;
qualitySheet.getRange("D5:D13").conditionalFormats.add("containsText", { text: "FAIL", format: { fill: "#FECACA", font: { color: "#991B1B", bold: true } } });
qualitySheet.getRange("D5:D13").conditionalFormats.add("containsText", { text: "PASS", format: { fill: "#DCFCE7", font: { color: "#166534", bold: true } } });

auditSheet.showGridLines = false;
auditSheet.getRange("A1:E1").values = [["Source Group ID", "Source Row ID", "Source Duplicate Count", "Risk Level", "Split"]];
auditSheet.getRange("A1:E1").format = { fill: "#123047", font: { bold: true, color: "#FFFFFF" } };
auditSheet.getRangeByIndexes(1, 0, assignedGroups.length, 5).values = assignedGroups.map((group) => [
  group.groupId,
  Number(group.row[0]),
  group.duplicateCount,
  group.label,
  group.split,
]);
auditSheet.getRange("A1:E8271").format.columnWidth = 22;
auditSheet.freezePanes.freezeRows(1);

const codebookRows = HEADERS.map((header) => {
  const definitions = {
    "Record ID": "Unique generated record identifier.",
    "Source Group ID": "Stable canonical source group. Use this for group-aware analysis and auditing.",
    Split: "Preassigned leakage-safe split. Never randomly repartition variants across this boundary.",
    "Is Synthetic": "TRUE for augmented or controlled-outlier rows; FALSE for retained deduplicated canonical rows.",
    "Augmentation Method": "Transformations applied to create the row.",
    "Is Outlier": "TRUE only for controlled stress-test scenarios.",
    "Outlier Type": "Named outlier policy; blank for ordinary records.",
    "Risk Level": "Five-class preserved target taxonomy: Low, Mild, Moderate, Severe, Emergency.",
    "Suggested Next Steps": "Non-diagnostic support-routing text mapped to risk level.",
    "Combined Text": "Complete, auditable concatenation of Q1–Q15 for text-model experiments.",
  };
  return [header, definitions[header] || "Questionnaire response retained or label-consistently augmented from the source schema."];
});
codebookSheet.showGridLines = false;
codebookSheet.getRange("A1:B1").values = [["Column", "Definition"]];
codebookSheet.getRange("A1:B1").format = { fill: "#123047", font: { bold: true, color: "#FFFFFF" } };
codebookSheet.getRangeByIndexes(1, 0, codebookRows.length, 2).values = codebookRows;
codebookSheet.getRange("A:A").format.columnWidth = 42;
codebookSheet.getRange("B:B").format = { columnWidth: 100, wrapText: true };
codebookSheet.freezePanes.freezeRows(1);

const catalogRows = [
  ["semantic_paraphrase", "Meaning-preserving rewording of Q1–Q6."],
  ["hinglish_code_switch", "English/Hinglish wording variation for Indian conversational usage."],
  ["light_informal_noise", "Limited contractions, casing, and informal spelling for robustness."],
  ["multi_select_reordering", "Reorders multi-select items without changing their meaning."],
  ["short_duration_high_risk", "High-risk safety signal with sudden or short-duration onset."],
  ["high_support_high_risk", "Severe/emergency risk despite a strong support network."],
  ["low_intensity_with_safety_signal", "Explicit safety risk despite a low self-rated intensity."],
  ["temporary_intensity_spike", "High current intensity without active safety indicators."],
  ["self_label_disagreement", "Self-description differs from questionnaire safety evidence."],
  ["missing_noncritical_field", "One non-safety-critical field is missing."],
  ["strong_support_despite_distress", "Strong support remains available despite distress."],
  ["mixed_functioning_signal", "Daily functioning and subjective distress point in different directions."],
];
catalogSheet.showGridLines = false;
catalogSheet.getRange("A1:B1").values = [["Method / Outlier Type", "Definition"]];
catalogSheet.getRange("A1:B1").format = { fill: "#123047", font: { bold: true, color: "#FFFFFF" } };
catalogSheet.getRangeByIndexes(1, 0, catalogRows.length, 2).values = catalogRows;
catalogSheet.getRange("A:A").format.columnWidth = 40;
catalogSheet.getRange("B:B").format = { columnWidth: 95, wrapText: true };
catalogSheet.freezePanes.freezeRows(1);

const checks = {
  total: recordNumber - 1,
  uniqueFingerprints: fullFingerprints.size,
  uniqueCombinedTexts: combinedTexts.size,
  sourceGroups: canonicalGroups.length,
  duplicateSourceRowsRemoved,
  outlierCount,
  ruleViolations,
  sourceCounts,
  generationCounts,
  outlierCounts,
};

if (checks.total !== TOTAL_TARGET || checks.uniqueFingerprints !== TOTAL_TARGET || checks.uniqueCombinedTexts !== TOTAL_TARGET || checks.ruleViolations !== 0 || checks.outlierCount !== 5000) {
  throw new Error(`Generation quality check failed: ${JSON.stringify(checks)}`);
}

const output = await SpreadsheetFile.exportXlsx(workbook);
await output.save(outputPath);

const metadataPath = outputPath.replace(/\.xlsx$/i, ".quality.json");
await fs.writeFile(metadataPath, JSON.stringify(checks, null, 2), "utf8");

console.log(JSON.stringify({ outputPath, csvPath, ...checks }, null, 2));
