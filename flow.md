# RAG Pipeline Flow (This Project)

This document explains the end-to-end RAG flow implemented in this repository for Amazon-style review analysis.

## 1) Big Picture

There are two major phases:

1. **Ingestion phase**: take review CSV data, turn each review into an embedding, and store vectors in Pinecone.
2. **Query phase (RAG)**: take a user question (or incoming review), retrieve similar stored reviews from Pinecone, and ask the LLM to answer using that retrieved context.

---

## 2) Route Entry Points

All routes are registered in `src/routes/index.js`.

- `POST /ingest-csv`  
  JSON body with `csvPath` (and optional column overrides) to ingest a CSV from disk.

- `POST /ingest-csv/upload`  
  Multipart upload with file field name `file` to ingest CSV directly from a request.

- `POST /rag`  
  JSON-only body with `message` (or aliases like `query`) and optional `k` to run retrieval + generation.

---

## 3) Ingestion Flow (CSV -> Pinecone)

### Step A: Parse input CSV

Implemented in `src/routes/routeHandlers.js`:

- `handleIngestCsvJson` for path-based CSV (`/ingest-csv`)
- `handleIngestCsvUpload` for multipart file (`/ingest-csv/upload`)

Both paths call helpers in `src/ingest/csvReviews.js`:

- `parseAmazonReviewsCsv(...)` or `readReviewsCsv(...)`
- `rowsToVectorRecords(...)` to convert rows into normalized records

Each normalized record contains:

- `id` (stable hash-based id)
- `text` (review text used for embedding)
- `metadata` (e.g., `asin`, `rating`, short text copy, row index)

### Step B: Create embeddings + upsert

Implemented in `src/pinecone/upsertReviews.js`:

1. Batch review texts.
2. Call `embedDocuments(...)` from `src/embeddings/generate.js`.
3. Build Pinecone payload: `{ id, values, metadata }`.
4. Call `index.upsert(...)` using `src/pinecone/client.js`.

Important: embedding dimension must match Pinecone index dimension (`EMBEDDING_DIMENSIONS` in env).

---

## 4) Query/RAG Flow (`POST /rag`)

Handled by `handleRag` in `src/routes/routeHandlers.js`.

### Step A: Read request body

Expected JSON:

```json
{
  "message": "What do customers complain about most in these reviews?",
  "k": 8
}
```

Accepted text keys include `message`, `query`, `input`, `prompt`, `question`.

### Step B: Run RAG pipeline

`handleRag` calls `runAmazonReviewRag(...)` in `src/rag/pipeline.js`.

Inside `runAmazonReviewRag`:

1. **Embed user message**  
   `embedText(trimmedMessage)` -> query vector

2. **Retrieve from Pinecone**  
   `index.query({ vector, topK, includeMetadata: true })`

3. **Build context block**  
   Format top matches with id, score, optional rating/asin, and review text from metadata.

4. **Generate answer with LLM**  
   `ChatOpenAI.invoke([...])` with:
   - system prompt (default analyst instructions, or caller override)
   - human prompt containing retrieved context + user message

5. **Return output**  
   JSON includes:
   - `answer` (LLM final response)
   - `sources` (id, score, rating, asin, textPreview)

---

## 5) Why This Is RAG

It follows classic RAG architecture:

- **Retrieval**: semantic nearest-neighbor search over stored vectors in Pinecone.
- **Augmentation**: inject retrieved review snippets into the prompt context.
- **Generation**: LLM generates grounded output based on retrieved snippets.

This reduces hallucination risk versus raw LLM-only answering, because the model is prompted with relevant review evidence first.

---

## 6) Operational Notes / Debugging

- Ensure `POST /rag` uses `Content-Type: application/json`.
- Confirm index dimension == embedding dimension.
- Use `GET /routes` to verify route contract quickly.
- Set `DEBUG_HTTP=1` to log request method/path/content-type (`src/routes/index.js`).
- If retrieval quality is poor:
  - verify ingest succeeded and vectors were upserted
  - increase/decrease `k`
  - inspect `sources` scores and previews

---

## 7) Mental Model (Python Analogy)

Think of the system like this:

1. Build a vectorized dataframe once (ingest step).
2. For each query:
   - embed query
   - do nearest-neighbor lookup
   - pass top rows into an LLM prompt template
   - return model output + cited rows

That is exactly what this Node.js implementation is doing, just with Express + Pinecone + LangChain JS.

