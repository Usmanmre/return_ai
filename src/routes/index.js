import express from "express";
import { config } from "../utils/config.js";
import {
  csvFileUploadMiddleware,
  handleIngestCsvUpload,
  handleIngestCsvJson,
  handleLocalIngest,
  handleLocalSearch,
  handleRag,
} from "./routeHandlers.js";

const DEBUG_HTTP = process.env.DEBUG_HTTP === "1";

/** All HTTP routes are registered here — open this file when debugging paths. */
export function registerRoutes(app) {
  if (DEBUG_HTTP) {
    app.use((req, _res, next) => {
      const ct = req.headers["content-type"] || "";
      console.log(
        `[HTTP] ${req.method} ${req.originalUrl} content-type=${JSON.stringify(ct)}`
      );
      next();
    });
  }

  // --- Multipart (must run before express.json — stream consumed once) ---
  app.post(
    "/ingest-csv/upload",
    csvFileUploadMiddleware,
    handleIngestCsvUpload
  );
 

  app.use(express.json({ limit: "1mb" }));

  // --- Discovery ---
  app.get("/", (_req, res) => {
    res.json({
      ok: true,
      routesFile: "src/routes/index.js",
      post: [
        "/ingest",
        "/search",
        "/ingest-csv",
        "/ingest-csv/upload (multipart, field: file)",
        "/rag",
      ],
    });
  });

  app.get("/routes", (_req, res) => {
    res.json({
      ok: true,
      description: "Static list; see src/routes/index.js for middleware order.",
      get: ["/", "/routes", "/health"],
      post: [
        { path: "/ingest", body: "JSON optional path, chunkSize, chunkOverlap" },
        { path: "/search", body: 'JSON { "query", "k" }' },
        { path: "/ingest-csv", body: 'JSON { "csvPath", ... }' },
        {
          path: "/ingest-csv/upload",
          body: "multipart field file + optional text fields",
        },
        {
          path: "/rag",
          body: 'JSON only: { "message", "k" } — Content-Type: application/json',
        },
      ],
    });
  });

  // --- Local in-memory demo ---
  app.post("/ingest", handleLocalIngest);
  app.post("/search", handleLocalSearch);

  // --- Amazon CSV + Pinecone + RAG ---
  app.post("/ingest-csv", handleIngestCsvJson);
  app.post("/rag", handleRag);

  app.get("/health", (_req, res) => {
    res.json({ ok: true });
  });
}
