import express from "express";
import { config } from "./src/utils/config.js";
import { registerRoutes } from "./src/routes/index.js";

const app = express();
registerRoutes(app);

app.listen(config.port, () => {
  console.log(`Server listening on http://localhost:${config.port}`);
  console.log(
    `Embedding dimensions (OpenAI → Pinecone): ${config.embeddingDimensions} — must match your Pinecone index`
  );
  console.log("Routes: see src/routes/index.js (set DEBUG_HTTP=1 to log each request)");
  console.log("GET  /  or GET /routes — list endpoints");
  console.log("--- Local demo (in-memory) ---");
  console.log("POST /ingest  — sample.txt → chunks → memory");
  console.log('POST /search — body: { "query": "...", "k": 5 }');
  console.log("--- Amazon reviews + Pinecone + RAG ---");
  console.log('POST /ingest-csv — JSON: { "csvPath": "data/file.csv" }');
  console.log('POST /ingest-csv/upload — multipart field "file" (CSV)');
  console.log('POST /rag — application/json only: { "message": "...", "k": 8 }');
});
