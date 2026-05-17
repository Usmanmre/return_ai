import { ChatOpenAI } from "@langchain/openai";
import { embedText } from "../embeddings/generate.js";
import { config } from "../utils/config.js";
import { getPineconeIndex } from "../pinecone/client.js";
import { Client } from "@elastic/elasticsearch";

// Use the client version that matches the server where possible.
// If you run Elasticsearch 8.x, prefer installing @elastic/elasticsearch v8.
// For demo setups that already have a compatible client/server pair, the
// default client headers will be correct and no manual header override is needed.
const es = new Client({
  node: process.env.ELASTIC_URL || "http://localhost:9200",
});

const DEFAULT_SYSTEM = `You are an analyst for E-commerce store product reviews.
You will receive:
1) CONTEXT: snippets of historical reviews retrieved by semantic similarity.
2) USER MESSAGE: a question, or a new review to classify / compare / explain.

Rules:
- Ground your answer in the CONTEXT when it is relevant. If the context does not contain enough evidence, say so.
- Do not invent specific star ratings or claims not supported by the context.
- Be concise and structured when helpful (bullets for themes).`;

/**
 * RAG: embed user message → Pinecone similarity search → LLM answer.
 */
export async function runAmazonReviewRag({
  userMessage,
  topK = config.ragDefaultTopK,
  systemPrompt,
}) {
  const system =
    typeof systemPrompt === "string" && systemPrompt.trim()
      ? systemPrompt.trim()
      : DEFAULT_SYSTEM;
  const trimmed = (userMessage || "").trim();
  if (!trimmed) throw new Error("userMessage is required");

  const queryVector = await embedText(trimmed);
  const index = getPineconeIndex();
// createReviewsIndex() 
 const [vectorResponse, bm25Results] = await Promise.all([
  index.query({
    vector: queryVector,
    topK,
    includeMetadata: true,
  }),
  bm25Search(trimmed, topK)
]);

const vectorMatches = vectorResponse.matches || [];
const bm25Matches = bm25Results || [];
const matches = mergeResults(vectorMatches, bm25Matches);

  const blocks = matches
    .filter((m) => (m.score ?? 0) >= 0.3)
    .map((m, i) => {
      const text = m.metadata?.text ? String(m.metadata.text) : "";
      const rating = m.metadata?.rating;
      const asin = m.metadata?.asin;
      const head = `[#${i + 1} id=${m.id} score=${(m.score ?? 0).toFixed(4)}${rating != null ? ` rating=${rating}` : ""}${asin ? ` asin=${asin}` : ""}]`;
      return `${head}\n${text}`;
    })
    .filter((b) => b.length > 0);

  const contextBlock =
    blocks.length > 0
      ? blocks.join("\n\n---\n\n")
      : "(No retrieved reviews — index may be empty or unrelated.)";

  console.log(`[rag] retrieved ${matches.length} Pinecone matches (topK=${topK})`);

  const llm = new ChatOpenAI({
    openAIApiKey: config.openaiApiKey,
    modelName: config.openaiChatModel,
    temperature: 0.2,
  });


  const response = await llm.invoke([
    ["system", system],
    [
      "human",
      `CONTEXT (retrieved reviews):\n${contextBlock}\n\n---\nUSER MESSAGE:\n${trimmed}`,
    ],
  ]);

  const answer =
    typeof response.content === "string"
      ? response.content
      : JSON.stringify(response.content);

  const sources = matches.map((m) => ({
    id: m.id,
    score: m.score ?? null,
    rating: m.metadata?.rating ?? null,
    asin: m.metadata?.asin ? String(m.metadata.asin) : null,
    textPreview: m.metadata?.text
      ? String(m.metadata.text).slice(0, 240)
      : null,
  }));

  return { answer, sources };
}

async function bm25Search(query, topK) {
  console.log('[rag] checking for existing "reviews" index in Elasticsearch...');
  try {
    const res = await es.search({
      index: "reviews",
      size: topK,
      query: {
        match: {
          text: query,
        },
      },
    });

    return (res.hits.hits || []).map((h) => ({
      id: h._id,
      score: h._score,
      text: h._source.text,
      rating: h._source.rating,
      asin: h._source.asin,
    }));
  } catch (err) {
    // If the index doesn't exist, return an empty result list and log a helpful message.
    const errType = err?.meta?.body?.error?.type;
    if (errType === "index_not_found_exception") {
      console.warn(
        '[rag] Elasticsearch index "reviews" not found — BM25 disabled. Create the index to enable BM25.'
      );
      return [];
    }
    throw err;
  }
}

/**
 * Create a minimal `reviews` index mapping (text + optional numeric fields).
 * Call this once to create the index if you don't want to use curl / Kibana.
 */
export async function createReviewsIndex() {
  console.log('[rag] checking for existing "reviews" index in Elasticsearch...');
  const exists = await es.indices.exists({ index: "reviews" });
  if (exists.body === true) {
    console.log('[rag] reviews index already exists');
    return { created: false };
  }

  const body = {
    mappings: {
      properties: {
        text: { type: "text" },
        rating: { type: "float" },
        asin: { type: "keyword" },
      },
    },
  };

  const res = await es.indices.create({ index: "reviews", body });
  return res.body || res;
}

function rrf(rank, k = 6) {
  return 1 / (k + rank);
}

function mergeResults(vectorMatches, bm25Matches) {
  const map = new Map();

  vectorMatches.forEach((m, i) => {
    map.set(m.id, (map.get(m.id) || 0) + rrf(i));
  });

  bm25Matches.forEach((m, i) => {
    map.set(m.id, (map.get(m.id) || 0) + rrf(i));
  });

  return [...map.entries()]
    .sort((a, b) => b[1] - a[1])
    .map(([id]) => {
      return (
        vectorMatches.find((v) => v.id === id) ||
        bm25Matches.find((b) => b.id === id)
      );
    })
    .filter(Boolean);
}