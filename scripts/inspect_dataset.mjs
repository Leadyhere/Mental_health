import { FileBlob, SpreadsheetFile } from "@oai/artifact-tool";

const datasetPath = process.argv[2];
if (!datasetPath) throw new Error("Dataset path is required");

const input = await FileBlob.load(datasetPath);
const workbook = await SpreadsheetFile.importXlsx(input);

const sheets = await workbook.inspect({
  kind: "sheet",
  include: "id,name",
  maxChars: 4000,
});
console.log("SHEETS");
console.log(sheets.ndjson);

const overview = await workbook.inspect({
  kind: "workbook,sheet,table,region",
  maxChars: 12000,
  tableMaxRows: 8,
  tableMaxCols: 30,
  tableMaxCellChars: 120,
});
console.log("OVERVIEW");
console.log(overview.ndjson);
