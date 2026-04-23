import fs from "fs/promises";
import path from "path";
import { fileURLToPath } from "url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const PROJECT_ROOT = path.resolve(__dirname, "../..");

/**
 * Split text into overlapping character windows.
 * chunk_size: max characters per chunk
 * chunk_overlap: characters shared between consecutive chunks (keeps context across boundaries)
 */
export function chunkText(text, { chunkSize, chunkOverlap }) {
  if (chunkSize <= 0) throw new Error("chunkSize must be positive");
  if (chunkOverlap < 0) throw new Error("chunkOverlap must be >= 0");
  if (chunkOverlap >= chunkSize) {
    throw new Error("chunkOverlap should be smaller than chunkSize");
  }

  const normalized = text.replace(/\r\n/g, "\n").trim();
  if (!normalized) return [];

  const chunks = [];
  let start = 0;

  while (start < normalized.length) {
    const end = Math.min(start + chunkSize, normalized.length);
    chunks.push(normalized.slice(start, end));
    if (end === normalized.length) break;
    start = end - chunkOverlap;
  }

  return chunks;
}

/**
 * Read data/sample.txt (or custom path relative to project root) and return chunks.
 */
export async function loadAndChunk(
  relativePath = "data/sample.txt",
  { chunkSize, chunkOverlap }
) {
  const filePath = path.join(PROJECT_ROOT, relativePath);
  const raw = await fs.readFile(filePath, "utf8");
  const chunks = chunkText(raw, { chunkSize, chunkOverlap });
  return { filePath, chunks };
}
