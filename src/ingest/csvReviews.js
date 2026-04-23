import fs from "fs/promises";
import path from "path";
import { parse } from "csv-parse/sync";
import { fileURLToPath } from "url";
import { createHash } from "crypto";
import { config } from "../utils/config.js";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const PROJECT_ROOT = path.resolve(__dirname, "../..");

const TEXT_COLUMN_CANDIDATES = [
  "reviewText",
  "review_text",
  "text",
  "body",
  "review",
  "comment",
  "content",
];

/**
 * Pick the main review text column from CSV headers.
 */
export function detectTextColumn(headers, explicit) {
  if (explicit) {
    const found = headers.find(
      (h) => h.trim().toLowerCase() === String(explicit).trim().toLowerCase()
    );
    if (found) return found;
  }
  const lower = headers.map((h) => h.trim());
  for (const cand of TEXT_COLUMN_CANDIDATES) {
    const idx = lower.findIndex((h) => h.toLowerCase() === cand.toLowerCase());
    if (idx >= 0) return headers[idx];
  }
  throw new Error(
    `Could not detect review text column. Headers: ${headers.join(", ")}. ` +
      `Pass textColumn in the request (e.g. "reviewText").`
  );
}

function pickColumn(headers, explicit, candidates) {
  if (explicit) {
    const found = headers.find(
      (h) => h.trim().toLowerCase() === String(explicit).trim().toLowerCase()
    );
    if (found) return found;
  }
  const lower = headers.map((h) => h.trim());
  for (const cand of candidates) {
    const idx = lower.findIndex((h) => h.toLowerCase() === cand.toLowerCase());
    if (idx >= 0) return headers[idx];
  }
  return null;
}

/**
 * Parse CSV buffer/string into row objects (first line = headers).
 */
function stripBomKeys(row) {
  const out = {};
  for (const [k, v] of Object.entries(row)) {
    out[k.replace(/^\uFEFF/, "")] = v;
  }
  return out;
}

export function parseAmazonReviewsCsv(content) {
  const records = parse(content, {
    columns: true,
    skip_empty_lines: true,
    trim: true,
    relax_column_count: true,
  }).map(stripBomKeys);
  if (!records.length) return { records: [], headers: [] };
  const headers = Object.keys(records[0]);
  return { records, headers };
}

/**
 * Read CSV from project-relative path.
 */
export async function readReviewsCsv(relativePath) {
  const filePath = path.join(PROJECT_ROOT, relativePath);
  const raw = await fs.readFile(filePath, "utf8");
  const { records, headers } = parseAmazonReviewsCsv(raw);
  return { filePath, records, headers };
}

function stableId(rowIndex, text) {
  const h = createHash("sha256").update(`${rowIndex}|${text}`).digest("hex");
  return `rev-${h.slice(0, 32)}`;
}

function truncate(str, max) {
  if (!str || str.length <= max) return str;
  return str.slice(0, max) + "…";
}

/**
 * Build Pinecone-ready rows: id, text (for embedding), metadata (Pinecone-safe).
 */
export function rowsToVectorRecords(
  records,
  headers,
  {
    textColumn,
    summaryColumn,
    ratingColumn,
    maxRows,
    maxTextChars = config.maxMetadataTextChars,
  }
) {
  const textKey = detectTextColumn(headers, textColumn);
  const summaryKey = summaryColumn
    ? pickColumn(headers, summaryColumn, ["summary", "review_summary", "title"])
    : pickColumn(headers, null, ["summary", "review_summary", "title"]);
  const ratingKey = ratingColumn
    ? pickColumn(headers, ratingColumn, [])
    : pickColumn(headers, null, [
        "overall",
        "rating",
        "stars",
        "score",
        "star_rating",
      ]);

  const slice = maxRows ? records.slice(0, maxRows) : records;
  const out = [];

  for (let i = 0; i < slice.length; i++) {
    const row = slice[i];
    const text = (row[textKey] || "").trim();
    if (!text) continue;

    const summary = summaryKey ? String(row[summaryKey] || "").trim() : "";
    const rating =
      ratingKey && row[ratingKey] !== undefined && row[ratingKey] !== ""
        ? Number(row[ratingKey])
        : undefined;

    const meta = {
      text: truncate(text, maxTextChars),
      text_column: textKey,
      row_index: String(i),
    };
    if (summary) meta.summary = truncate(summary, 2000);
    if (Number.isFinite(rating)) meta.rating = rating;

    const asinKey = pickColumn(Object.keys(row), null, ["asin", "productId", "product_id"]);
    if (asinKey && row[asinKey]) meta.asin = String(row[asinKey]).trim();

    out.push({
      id: stableId(i, text),
      text,
      metadata: meta,
    });
  }

  return { textKey, records: out, skippedEmpty: slice.length - out.length };
}
