#!/usr/bin/env node
/**
 * CLI: POST /search against a running local server.
 * Usage: node testSearch.js "your query here" [k]
 *
 * Prerequisite: server running and POST /ingest completed once.
 */

const PORT = process.env.PORT || 3000;
const BASE = `http://127.0.0.1:${PORT}`;

const query = process.argv[2];
const k = process.argv[3] ? parseInt(process.argv[3], 10) : 5;

if (!query) {
  console.error('Usage: node testSearch.js "your query here" [k]');
  process.exit(1);
}

const res = await fetch(`${BASE}/search`, {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({ query, k }),
});

const data = await res.json().catch(() => ({}));

if (!res.ok) {
  console.error("Request failed:", res.status, data);
  process.exit(1);
}

console.log(`Query: ${data.query}`);
console.log(`k: ${data.k}`);
console.log("--- ranked results ---");
for (const r of data.results || []) {
  console.log(`score=${r.score.toFixed(6)}  id=${r.id}`);
  console.log(r.text);
  console.log("");
}
