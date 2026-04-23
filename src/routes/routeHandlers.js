import multer from "multer";
import { config } from "../utils/config.js";
import { loadAndChunk } from "../ingest/loadAndChunk.js";
import {
  embedDocuments,
  logEmbeddingPreview,
} from "../embeddings/generate.js";
import { clearStore, addRecords, count } from "../vectorstore/memoryStore.js";
import { searchTopK } from "../search/retrieve.js";
import {
  parseAmazonReviewsCsv,
  readReviewsCsv,
  rowsToVectorRecords,
} from "../ingest/csvReviews.js";
import { embedAndUpsertReviews } from "../pinecone/upsertReviews.js";
import { runAmazonReviewRag } from "../rag/pipeline.js";

const upload = multer({
  storage: multer.memoryStorage(),
  limits: { fileSize: 52 * 1024 * 1024 },
});

/** Multipart CSV upload: field name `file`. */
export const csvFileUploadMiddleware = upload.single("file");

export async function handleIngestCsvUpload(req, res) {
  try {
    if (!req.file?.buffer) {
      return res.status(400).json({
        ok: false,
        error:
          'Missing CSV file field "file" (multipart/form-data, field name: file)',
      });
    }

    const { records, headers } = parseAmazonReviewsCsv(
      req.file.buffer.toString("utf8")
    );
    if (!records.length) {
      return res.status(400).json({
        ok: false,
        error: "CSV has no data rows",
      });
    }

    const body = req.body || {};
    const { textKey, records: vectorRecords, skippedEmpty } =
      rowsToVectorRecords(records, headers, {
        textColumn: body.textColumn,
        summaryColumn: body.summaryColumn,
        ratingColumn: body.ratingColumn,
        maxRows: body.maxRows != null ? Number(body.maxRows) : undefined,
      });

    if (!vectorRecords.length) {
      return res.status(400).json({
        ok: false,
        error: "No non-empty review texts found in CSV",
      });
    }

    console.log(
      `\n[ingest-csv/upload] filename=${req.file.originalname} rows=${records.length} embed=${vectorRecords.length} skippedEmpty=${skippedEmpty} textColumn=${textKey}`
    );

    const { upserted } = await embedAndUpsertReviews(vectorRecords);

    res.json({
      ok: true,
      originalName: req.file.originalname,
      textColumn: textKey,
      rowsInCsv: records.length,
      vectorsPrepared: vectorRecords.length,
      skippedEmpty,
      pineconeUpserted: upserted,
    });
  } catch (err) {
    console.error("[ingest-csv/upload] error:", err);
    res.status(500).json({
      ok: false,
      error: err instanceof Error ? err.message : String(err),
    });
  }
}

export async function handleLocalIngest(req, res) {
  try {
    const body = req.body && typeof req.body === "object" ? req.body : {};
    const relPath =
      typeof body.path === "string" && body.path.length
        ? body.path
        : "data/sample.txt";
    const chunkSize =
      Number(body.chunkSize) > 0 ? Number(body.chunkSize) : config.chunkSize;
    const chunkOverlap =
      Number(body.chunkOverlap) >= 0
        ? Number(body.chunkOverlap)
        : config.chunkOverlap;

    const { filePath, chunks } = await loadAndChunk(relPath, {
      chunkSize,
      chunkOverlap,
    });

    console.log("\n========== INGEST: CHUNKS ==========");
    console.log(`file: ${filePath}`);
    console.log(`chunk_size=${chunkSize} chunk_overlap=${chunkOverlap}`);
    chunks.forEach((c, i) => {
      console.log(`--- chunk ${i} (${c.length} chars) ---\n${c}\n`);
    });
    console.log("====================================\n");

    clearStore();
    const vectors = await embedDocuments(chunks);

    if (vectors.length !== chunks.length) {
      throw new Error("Embedding count does not match chunk count");
    }

    vectors.forEach((vec, i) => {
      logEmbeddingPreview(`document chunk ${i}`, vec);
    });

    const newRecords = chunks.map((text, i) => ({
      id: `chunk-${i}`,
      text,
      embedding: vectors[i],
      metadata: {
        source: relPath,
        chunkIndex: i,
        chunkSize,
        chunkOverlap,
      },
    }));

    addRecords(newRecords);

    res.json({
      ok: true,
      source: relPath,
      chunks: chunks.length,
      stored: count(),
    });
  } catch (err) {
    console.error("[ingest] error:", err);
    res.status(500).json({
      ok: false,
      error: err instanceof Error ? err.message : String(err),
    });
  }
}

export async function handleLocalSearch(req, res) {
  try {
    const body = req.body && typeof req.body === "object" ? req.body : {};
    const query = typeof body.query === "string" ? body.query.trim() : "";
    const k = Number(body.k) > 0 ? Number(body.k) : 5;

    if (!query) {
      return res.status(400).json({ ok: false, error: "query is required" });
    }

    if (count() === 0) {
      return res.status(400).json({
        ok: false,
        error: "Vector store is empty. POST /ingest first.",
      });
    }

    console.log("\n========== SEARCH ==========");
    console.log(`query: ${query}`);
    console.log(`k: ${k}`);
    const results = await searchTopK(query, k);
    console.log("============================\n");

    res.json({ ok: true, query, k, results });
  } catch (err) {
    console.error("[search] error:", err);
    res.status(500).json({
      ok: false,
      error: err instanceof Error ? err.message : String(err),
    });
  }
}

export async function handleIngestCsvJson(req, res) {
  try {
    const body = req.body && typeof req.body === "object" ? req.body : {};
    const csvPath =
      typeof body.csvPath === "string" && body.csvPath.trim()
        ? body.csvPath.trim()
        : "";

    if (!csvPath) {
      return res.status(400).json({
        ok: false,
        error:
          'Missing csvPath. Example: { "csvPath": "data/amazon_reviews_sample.csv" }',
      });
    }

    const { filePath, records, headers } = await readReviewsCsv(csvPath);
    if (!records.length) {
      return res.status(400).json({
        ok: false,
        error: "CSV has no data rows",
      });
    }

    const { textKey, records: vectorRecords, skippedEmpty } =
      rowsToVectorRecords(records, headers, {
        textColumn: body.textColumn,
        summaryColumn: body.summaryColumn,
        ratingColumn: body.ratingColumn,
        maxRows: body.maxRows != null ? Number(body.maxRows) : undefined,
      });

    if (!vectorRecords.length) {
      return res.status(400).json({
        ok: false,
        error: "No non-empty review texts found in CSV",
      });
    }

    console.log(
      `\n[ingest-csv] file=${filePath} rows=${records.length} embed=${vectorRecords.length} skippedEmpty=${skippedEmpty} textColumn=${textKey}`
    );

    const { upserted } = await embedAndUpsertReviews(vectorRecords);

    res.json({
      ok: true,
      filePath,
      textColumn: textKey,
      rowsInCsv: records.length,
      vectorsPrepared: vectorRecords.length,
      skippedEmpty,
      pineconeUpserted: upserted,
    });
  } catch (err) {
    console.error("[ingest-csv] error:", err);
    res.status(500).json({
      ok: false,
      error: err instanceof Error ? err.message : String(err),
    });
  }
}

function pickRagUserText(body) {
  const keys = [
    "message",
    "query",
    "input",
    "prompt",
    "question",
    "userMessage",
    "user_message",
  ];
  for (const key of keys) {
    const v = body[key];
    if (typeof v === "string" && v.trim()) return v;
  }
  return "";
}

export async function handleRag(req, res) {
  try {
    const body = req.body && typeof req.body === "object" ? req.body : {};
    const message = pickRagUserText(body);
    const k =
      Number(body.k) > 0 ? Number(body.k) : config.ragDefaultTopK;
    const systemPrompt =
      typeof body.systemPrompt === "string" && body.systemPrompt.trim()
        ? body.systemPrompt.trim()
        : undefined;

    if (!message.trim()) {
      const keys = body && typeof body === "object" ? Object.keys(body) : [];
      const hint =
        keys.length === 0
          ? 'POST /rag is JSON-only. Use Content-Type: application/json, Body → raw → JSON (not form-data). Example: { "message": "...", "k": 8 }'
          : `Parsed JSON keys: ${keys.join(", ")}. Put your question in "message" (or "query" / "question").`;

      return res.status(400).json({
        ok: false,
        error:
          'Send a non-empty string in "message" (or "query", "input", "prompt", "question").',
        hint,
        contentType: req.headers["content-type"] || null,
        example: {
          message: "What do customers complain about most in these reviews?",
          k: 8,
        },
      });
    }

    const { answer, sources } = await runAmazonReviewRag({
      userMessage: message.trim(),
      topK: k,
      systemPrompt,
    });

    res.json({ ok: true, k, answer, sources });
  } catch (err) {
    console.error("[rag] error:", err);
    res.status(500).json({
      ok: false,
      error: err instanceof Error ? err.message : String(err),
    });
  }
}
