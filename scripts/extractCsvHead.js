#!/usr/bin/env node
/**
 * Stream the first N lines from a large CSV into a smaller file (memory-safe).
 *
 * Usage:
 *   node scripts/extractCsvHead.js
 *   node scripts/extractCsvHead.js test.csv data/test_first_1000.csv 1000
 *
 * Defaults: input test.csv (project root), output data/test_first_1000.csv, N=1000
 */

import fs from "fs";
import readline from "readline";
import path from "path";
import { fileURLToPath } from "url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const ROOT = path.resolve(__dirname, "..");

const inputPath = path.resolve(ROOT, process.argv[2] || "test.csv");
const outputPath = path.resolve(ROOT, process.argv[3] || "data/test_first_1000.csv");
const maxLines = Math.max(1, parseInt(process.argv[4] || "1000", 10) || 1000);

await fs.promises.mkdir(path.dirname(outputPath), { recursive: true });

const input = fs.createReadStream(inputPath, { encoding: "utf8" });
const rl = readline.createInterface({ input, crlfDelay: Infinity });
const out = fs.createWriteStream(outputPath, { encoding: "utf8" });

let written = 0;
for await (const line of rl) {
  out.write(line);
  out.write("\n");
  written++;
  if (written >= maxLines) break;
}

await new Promise((resolve, reject) => {
  out.end((err) => (err ? reject(err) : resolve()));
});

console.log(`Wrote ${written} lines to ${outputPath}`);
