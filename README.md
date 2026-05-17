# Vector embeddings, Pinecone, and RAG (Node.js)

**Node.js (ESM)** project with two layers:

1. **Learning path (local):** `POST /ingest` + `POST /search` — read `data/sample.txt`, chunk text, embed with OpenAI, store vectors in a **plain in-memory array**, rank with **hand-written cosine similarity** (no LLM).
2. **Amazon reviews path (production-shaped):** `POST /ingest-csv` (or `/ingest-csv/upload`) embeds each review row and **upserts to Pinecone**; `POST /rag` runs a **RAG pipeline** (Pinecone retrieval + **OpenAI chat** for the final answer). Use this for analyst questions or for **incoming review text** (same endpoint — you describe the task in natural language).

**LangChain** is used as a thin client for **`OpenAIEmbeddings`** and **`ChatOpenAI`** (`@langchain/openai`). The local demo still avoids LangChain vector-store abstractions; Pinecone is called via the **official `@pinecone-database/pinecone` client** (upsert/query), not LangChain’s vector wrappers.

---

## Table of contents

- [What you will learn](#what-you-will-learn)
- [Architecture (local demo)](#architecture)
- [Architecture: Amazon, Pinecone, RAG](#architecture-amazon-pinecone-rag)
- [Tech stack](#tech-stack)
- [Quick start](#quick-start)
- [Environment variables](#environment-variables)
- [Pinecone index setup](#pinecone-index-setup)
- [API reference](#api-reference)
- [How the pieces work](#how-the-pieces-work)
- [Testing (curl, Postman, CLI)](#testing-curl-postman-cli)
- [Project structure](#project-structure)
- [Interview Q&A (10+ questions)](#interview-qa-10-questions)
- [What this project deliberately does not do](#what-this-project-deliberately-does-not-do)
- [Optional extensions](#optional-extensions)

---

## What you will learn

| Topic | Where it lives |
|--------|----------------|
| Reading and normalizing text, then splitting into overlapping windows | `src/ingest/loadAndChunk.js` |
| Turning text into a dense vector via an API | `src/embeddings/generate.js` |
| Storing `{ id, text, embedding, metadata }` in memory | `src/vectorstore/memoryStore.js` |
| Cosine similarity by hand | `src/vectorstore/cosineSimilarity.js` |
| Query embedding + brute-force scan + sort + top‑k | `src/search/retrieve.js` |
| HTTP routes (single registration file) | `src/routes/index.js`, `server.js` |
| Parse Amazon-style CSV, detect review column | `src/ingest/csvReviews.js` |
| Upsert embeddings to Pinecone | `src/pinecone/upsertReviews.js`, `src/pinecone/client.js` |
| RAG: retrieve + LLM answer | `src/rag/pipeline.js`, `src/routes/routeHandlers.js` |

---

## Architecture

```text
data/sample.txt
      │
      ▼
 loadAndChunk  ──► chunks[] (strings)
      │
      ▼
 embedDocuments (OpenAI via LangChain)
      │
      ▼
 memoryStore   ──► [{ id, text, embedding, metadata }, ...]
      │
      ▼  (on search)
 embedQuery(query)  +  cosineSimilarity vs every row
      │
      ▼
 sort by score desc  →  slice(0, k)  →  JSON response
```

Search is **O(n)** in the number of stored vectors: fine for learning and small corpora; production systems use **approximate nearest neighbor** indexes when `n` is large.

---

## Architecture: Amazon, Pinecone, RAG

```text
reviews.csv  (rows: reviewText, overall, summary, asin, …)
      │
      ▼
 csvReviews.js  ──► one vector per review row (id + metadata.text truncated to Pinecone limits)
      │
      ▼
 embedDocuments (OpenAI)
      │
      ▼
 Pinecone upsert  (batched)
      │
      ▼  (on /rag)
 embedQuery(user message)  →  Pinecone query(topK)  →  build prompt with retrieved review texts
      │
      ▼
 ChatOpenAI  ──► final answer + source list (ids, scores, previews)
```

Each **CSV row** becomes **one embedding** (the full review text for that row). Very long reviews are truncated in **metadata** (`MAX_METADATA_TEXT_CHARS`) to stay within Pinecone metadata limits; adjust env or split long reviews in preprocessing if you need every token.

Note: CSV ingest currently embeds and upserts directly to Pinecone only. The project can optionally fuse BM25 search from Elasticsearch with Pinecone results in the RAG pipeline. To enable BM25 you must run an ES node and create a `reviews` index (see `src/rag/pipeline.js` → `createReviewsIndex()`), or implement a small sync that indexes CSV rows into ES during ingest.

---

## Tech stack

| Piece | Choice |
|--------|--------|
| Runtime | Node.js **18+** (native `fetch` in `testSearch.js`) |
| Modules | **ESM** (`"type": "module"` in `package.json`) |
| HTTP | **Express** |
| Config | **dotenv** (`src/utils/config.js`) |
| Embeddings + chat | **@langchain/openai** → `OpenAIEmbeddings`, `ChatOpenAI` |
| Local vector store | In-memory array + manual cosine (`src/vectorstore/`) |
| Hosted vector DB | **@pinecone-database/pinecone** (upsert / query) |
| CSV | **csv-parse** |
| Upload | **multer** (memory storage) |
| Search (BM25) | **Elasticsearch 8.x** (optional, used for BM25 fusion in RAG) |

---

## Quick start

```bash
cd /path/to/RAG_1
npm install
cp .env.example .env
# Required for all paths: OPENAI_API_KEY
# Required for CSV + RAG: PINECONE_API_KEY, PINECONE_INDEX_NAME (see Pinecone index setup)
npm run dev
```

Default base URL: `http://localhost:3000` (override with `PORT`).

Optional: Elasticsearch (BM25 fusion for `/rag`)

If you want the BM25 text search fused with Pinecone vectors during RAG, run an Elasticsearch 8.x node locally. A simple Docker example:

```bash
docker run -p 9200:9200 -p 9300:9300 \
     -e "discovery.type=single-node" \
     -e "xpack.security.enabled=false" \
     docker.elastic.co/elasticsearch/elasticsearch:8.12.0
```

The app expects an index named `reviews`. If the index is missing, the RAG pipeline will continue using Pinecone-only retrieval and log a warning. See `src/rag/pipeline.js` for a helper `createReviewsIndex()` that will create a minimal `reviews` mapping for you.

| Script | Command |
|--------|---------|
| Development (auto-restart) | `npm run dev` |
| Production-style | `npm start` |

---

## Environment variables

| Variable | Required | Default | Purpose |
|----------|----------|---------|---------|
| `OPENAI_API_KEY` | Yes | — | Embeddings + RAG chat |
| `OPENAI_EMBEDDING_MODEL` | No | `text-embedding-3-small` | Embedding model |
| `OPENAI_CHAT_MODEL` | No | `gpt-4o-mini` | Chat model for `/rag` |
| `EMBEDDING_DIMENSIONS` | No | `1536` | Vector length for `text-embedding-3*` (must match Pinecone index) |
| `PINECONE_API_KEY` | For CSV/RAG | — | Pinecone API key |
| `PINECONE_INDEX_NAME` | For CSV/RAG | — | Target index name |
| `PINECONE_HOST` | No | — | Index host URL if shown in Pinecone console |
| `PINECONE_UPSERT_BATCH` | No | `100` | Vectors per upsert batch (env name: PINECONE_UPSERT_BATCH) |
| `MAX_METADATA_TEXT_CHARS` | No | `32000` | Max stored review text per vector (metadata) |
| `RAG_DEFAULT_TOP_K` | No | `8` | Default `k` for Pinecone query in `/rag` |
| `PORT` | No | `3000` | HTTP port |
| `CHUNK_SIZE` | No | `400` | Local `/ingest` only |
| `CHUNK_OVERLAP` | No | `50` | Local `/ingest` only |

See `.env.example` for a template.

---

## Pinecone index setup

1. In the [Pinecone console](https://app.pinecone.io/), create a **serverless** index (or pod index) with:
   - **Metric:** `cosine` (matches typical semantic search with normalized embeddings).
   - **Dimensions:** `1536` if you use `text-embedding-3-small` with default size, or match whatever you set in `EMBEDDING_DIMENSIONS` for `text-embedding-3*` (index dimension **must** equal embedding vector length).
2. Copy the API key into `PINECONE_API_KEY` and the index name into `PINECONE_INDEX_NAME`.
3. If the console shows a **host** URL for the index, set `PINECONE_HOST` (some setups need it for `pc.index(name, host)`).

---

## API reference

### `GET /health`

Returns `{ "ok": true }` for uptime checks.

### `POST /ingest`

Loads a text file from the project, chunks it, embeds each chunk, **replaces** the in-memory store, and returns counts.

**Body (JSON, all optional):**

| Field | Type | Description |
|--------|------|-------------|
| `path` | string | Path relative to project root (default `data/sample.txt`) |
| `chunkSize` | number | Overrides `CHUNK_SIZE` when `> 0` |
| `chunkOverlap` | number | Overrides `CHUNK_OVERLAP` when `>= 0` |

**Success response:** `{ ok, source, chunks, stored }`

**Side effects:** Server logs each chunk and embedding previews (length + first values).

### `POST /search`

Embeds the query, scores every stored vector with cosine similarity, returns the top `k` results.

**Body (JSON):**

| Field | Type | Description |
|--------|------|-------------|
| `query` | string | **Required** (non-empty after trim) |
| `k` | number | Default `5`; must be `> 0` to override |

**Success response:** `{ ok, query, k, results }` where each result has `id`, `text`, `metadata`, `score`.

**Errors:** `400` if `query` is missing/empty or the store is empty (ingest first). `500` on server/embedding errors.

**Side effects:** Server logs query embedding preview and **all** similarity scores (sorted descending).

### `POST /ingest-csv` (Amazon CSV → Pinecone)

**Content-Type:** `application/json`

**Body:**

| Field | Type | Description |
|--------|------|-------------|
| `csvPath` | string | **Required.** Path under the project root, e.g. `data/amazon_reviews_sample.csv` |
| `textColumn` | string | Optional. CSV column for review body (auto-detects `reviewText`, `review_text`, `text`, …) |
| `summaryColumn` | string | Optional. Short summary column |
| `ratingColumn` | string | Optional. Star rating (auto-detects `overall`, `rating`, …) |
| `maxRows` | number | Optional. Ingest only the first N data rows (testing) |

**Behavior:** Parses rows, builds stable vector ids from content, embeds in batches, **upserts** into Pinecone (does not clear the whole index — re-upserting the same logical row overwrites that id).

### `POST /ingest-csv/upload`

**Content-Type:** `multipart/form-data`  
**Field:** `file` — the CSV file. Optional string fields: `textColumn`, `summaryColumn`, `ratingColumn`, `maxRows`.

### `POST /rag` (retrieval + final LLM output)

**Content-Type:** `application/json` only (raw JSON body — not `multipart` or `x-www-form-urlencoded`).

**Body:**

| Field | Type | Description |
|--------|------|-------------|
| `message` | string | **Required** (alias: `query`). Either an **analyst question** or **incoming review text** plus instructions (“Classify sentiment…”, “Compare to typical complaints…”). |
| `k` | number | Optional. Pinecone top‑K (default from `RAG_DEFAULT_TOP_K`). |
| `systemPrompt` | string | Optional. Override the default analyst system prompt. |

**Response:** `{ ok, k, answer, sources }` — `sources` lists Pinecone match ids, scores, and short text previews.

**Flow:** `src/rag/pipeline.js` embeds `message`, runs `index.query`, stitches retrieved `metadata.text` into the user prompt, calls `ChatOpenAI`.

Notes on BM25 / Elasticsearch:
- If an Elasticsearch `reviews` index is available, `runAmazonReviewRag` will fuse vector matches from Pinecone with BM25 hits from ES (reciprocal-rank fusion).
- If the `reviews` index is missing, the pipeline logs a warning and continues using Pinecone-only retrieval. Create the index with `createReviewsIndex()` (in `src/rag/pipeline.js`) or via curl/Kibana to enable BM25.

---

## How the pieces work

### Embeddings (brief)

An embedding model maps text into a **fixed-length vector** of floats. Vectors for **semantically similar** inputs tend to point in **similar directions** in that space, so geometric scores (here, cosine similarity) can approximate “closeness” of meaning. This demo does not train the model; it **calls** OpenAI’s embedding endpoint through LangChain.

### Chunking

`chunkText` in `src/ingest/loadAndChunk.js` uses **sliding character windows**: each chunk is at most `chunkSize` characters, and the next window starts at `end - chunkOverlap` so neighboring chunks share context. `chunkOverlap` must be **strictly less than** `chunkSize`.

### Cosine similarity

Implemented in `src/vectorstore/cosineSimilarity.js`:

\[
\cos(\theta) = \frac{\mathbf{a} \cdot \mathbf{b}}{\|\mathbf{a}\| \|\mathbf{b}\|}
\]

Higher values mean smaller angle between vectors (more alignment). Query and document embeddings **must** come from the **same model** (same dimensionality).

### Retrieval

`searchTopK` in `src/search/retrieve.js` embeds the query once, computes cosine similarity against **every** row, sorts by score descending, then returns the first `k` entries. Embeddings are **not** written back to the store during search.

---

## Testing (curl, Postman, CLI)

### 1. Ingest

```bash
curl -s -X POST http://localhost:3000/ingest \
  -H "Content-Type: application/json" \
  -d '{"chunkSize":400,"chunkOverlap":50}'
```

### 2. Search

```bash
curl -s -X POST http://localhost:3000/search \
  -H "Content-Type: application/json" \
  -d '{"query":"How do we measure vector similarity?","k":3}'
```

### Postman

- **POST** `http://localhost:3000/ingest` — Body: raw JSON `{}` or `{ "chunkSize": 300, "chunkOverlap": 40 }`
- **POST** `http://localhost:3000/search` — Body: `{ "query": "your text", "k": 5 }`

### CLI (`testSearch.js`)

Requires a running server and a completed ingest.

```bash
node testSearch.js "cosine similarity between vectors" 3
```

### 3. Amazon CSV → Pinecone (after index + env are ready)

```bash
curl -s -X POST http://localhost:3000/ingest-csv \
  -H "Content-Type: application/json" \
  -d '{"csvPath":"data/amazon_reviews_sample.csv","maxRows":100}'
```

### 4. RAG (question or new review)

```bash
curl -s -X POST http://localhost:3000/rag \
  -H "Content-Type: application/json" \
  -d '{"message":"What themes appear in negative reviews?","k":6}'
```

**Incoming review example:**

```bash
curl -s -X POST http://localhost:3000/rag \
  -H "Content-Type: application/json" \
  -d '{"message":"Incoming review: The hinge broke after a week. Is this aligned with past customer issues?","k":8}'
```

---

## Project structure

```text
.
├── server.js                 # Express entry, /health, mounts API routes
├── testSearch.js             # CLI client for POST /search
├── package.json
├── .env.example
├── data/
│   ├── sample.txt                 # Local /ingest demo
│   └── amazon_reviews_sample.csv  # Tiny CSV for /ingest-csv
└── src/
    ├── utils/
    │   └── config.js
    ├── ingest/
    │   ├── loadAndChunk.js
    │   └── csvReviews.js       # CSV parse + column detection + Pinecone metadata rows
    ├── embeddings/
    │   └── generate.js
    ├── vectorstore/
    │   ├── memoryStore.js
    │   └── cosineSimilarity.js
    ├── pinecone/
    │   ├── client.js
    │   └── upsertReviews.js
    ├── rag/
    │   └── pipeline.js         # Pinecone query + ChatOpenAI
    ├── search/
    │   └── retrieve.js
    └── routes/
        ├── index.js            # all route registrations (start here when debugging)
        └── routeHandlers.js    # POST/GET handler implementations
```

---

## Interview Q&A (13 questions)

Use this section to practice explaining the system in a technical interview. Answers are aligned with **this repository’s actual behavior**.

### 1. What problem does this project demonstrate?

**Answer:** It demonstrates the **retrieval** half of semantic search: turning documents and queries into vectors, storing them, and **ranking** stored chunks by **cosine similarity** to the query vector. It intentionally omits generation (no LLM) so the vector and scoring mechanics stay clear.

### 2. Why use chunks instead of embedding the entire file as one string?

**Answer:** A single huge string produces **one** vector, so you can only retrieve the **whole document** or nothing. Chunking gives **multiple vectors** per file, which improves **granularity**: the top match can be a **paragraph-sized** span that actually relates to the query. Trade-offs: chunk size affects context (too small loses meaning; too large dilutes specificity) and cost (more chunks ⇒ more embedding API calls).

### 3. What does `chunk_overlap` do in this codebase?

**Answer:** In `chunkText`, after each chunk the next window starts at `end - chunkOverlap` (`src/ingest/loadAndChunk.js`). Shared characters **bridge** chunk boundaries so concepts split across a cut are still partially present in adjacent chunks, which often helps retrieval quality.

### 4. Where are embeddings stored, and in what shape?

**Answer:** On the **local** path, each vector is a `number[]` kept in RAM inside `memoryStore.js` alongside `text` and `metadata`. On the **Amazon** path, **Pinecone** holds the vectors; this app sends `values` + `metadata` on upsert and only gets vectors back indirectly via similarity scores on query. In both cases the **same OpenAI embedding model** should be used for ingest and search.

### 5. Why does ingest call `clearStore()` before adding new records?

**Answer:** In `handleLocalIngest` (`src/routes/routeHandlers.js`), each successful `/ingest` **replaces** the corpus: `clearStore()` then `addRecords(...)`. That avoids mixing old and new embeddings from different files or different chunking settings, which would confuse debugging and metadata consistency.

### 6. Explain cosine similarity as implemented here and why it is common for embeddings.

**Answer:** `cosineSimilarity` in `src/vectorstore/cosineSimilarity.js` computes \(\sum_i a_i b_i\) divided by \(\|\mathbf{a}\|\|\mathbf{b}\|\). It measures the **angle** between vectors and is **scale-invariant** in the sense that multiplying a vector by a positive scalar does not change the cosine. Many embedding pipelines produce vectors where **direction** encodes meaning more than raw magnitude, so cosine is a standard choice.

### 7. What is the time complexity of search in this project?

**Answer:** **O(n · d)** for `n` stored vectors and `d` dimensions: for each of `n` rows we run one cosine similarity that loops over `d` dimensions (`retrieve.js` + `cosineSimilarity.js`). Sorting is **O(n log n)**. For large `n`, production systems use **ANN** (approximate nearest neighbor) indexes to avoid a full scan.

### 8. What happens if you mix embeddings from two different models?

**Answer:** `cosineSimilarity` throws if vector **lengths** differ (`src/vectorstore/cosineSimilarity.js`). Even if lengths matched by accident, comparing vectors from **different** models or training setups is usually **invalid**: the spaces are not aligned, so scores would not be meaningful.

### 9. Why is LangChain used if you avoid LangChain vector stores?

**Answer:** LangChain is a **thin client** for **`OpenAIEmbeddings`** (all embedding calls) and **`ChatOpenAI`** (the `/rag` completion). Vector **storage** is either a hand-written array + cosine (`memoryStore.js`, `cosineSimilarity.js`) or **Pinecone’s** own client — not LangChain’s `VectorStore` classes.

### 10. What is the difference between `embedDocuments` and `embedQuery` in practice?

**Answer:** In this repo, **documents** are embedded with `embedDocuments(chunks)` during ingest; the **query** uses `embedQuery` in `embedText` during search (`generate.js`, `retrieve.js`). Some APIs treat queries and documents symmetrically; OpenAI’s embedding endpoint can still apply slightly different preprocessing. Using the API’s **document** path for bulk chunks and **query** path for the user string matches common LangChain usage and keeps the demo faithful to the library’s intent.

### 11. How does the RAG path in this repo relate to the local demo?

**Answer:** The **local** `/search` path teaches brute-force cosine over an array. The **Amazon** path moves **storage and ANN search** to **Pinecone** (`src/pinecone/client.js`, upsert/query), then adds **generation** in `src/rag/pipeline.js`: retrieved review texts are concatenated into a prompt and `ChatOpenAI` produces the final answer. Same embedding model should be used for ingest and query so vectors stay comparable.

### 12. Why keep an in-memory path if Pinecone is the “real” store?

**Answer:** The in-memory route is still useful to **learn** similarity math without API keys to Pinecone, and to **debug** embeddings (logs, fixed small corpus). Pinecone is for **scale and persistence** when you load a large Amazon reviews CSV.

### 13. Why one embedding per CSV row instead of chunking each review?

**Answer:** Amazon review rows are usually **one natural document**; one vector per row keeps id ↔ review alignment simple and metadata (`asin`, `rating`, etc.) stays attached to that vector. If reviews are **very long**, you could split into sub-chunks with shared metadata (not implemented here) and upsert multiple vectors per row.

---

## What this project deliberately does not do

- **Local `/ingest` + `/search`** still do **not** call an LLM; they are retrieval-only for pedagogy.
- **No LangChain `VectorStore`** abstractions; Pinecone is used via the **official JS client**.
- **No auth** on HTTP endpoints — add API keys, OAuth, or network restrictions before exposing publicly.
- **No guarantee** that Pinecone metadata holds the **full** text of extremely long reviews (truncation); production systems sometimes store full text in S3/Postgres and keep only ids/snippets in Pinecone.

---

## Optional extensions

- **Namespaces / multi-tenant:** use Pinecone namespaces per product line or seller.
- **Hybrid search:** combine keyword (BM25) with vector hits for skewed vocabularies.
- **Chunk long reviews:** split on sentences, upsert multiple vectors sharing `asin` in metadata.
- **Alternative embeddings:** swap `OpenAIEmbeddings` in `generate.js`; **recreate** the Pinecone index to match new dimension and **re-ingest**.
- **Async jobs:** move CSV ingest to a queue worker for huge files.

---

## License

Use and modify freely for learning. Ensure you comply with **OpenAI’s** terms and pricing when calling their API.
