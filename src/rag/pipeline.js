import { ChatOpenAI } from "@langchain/openai";
import { embedText } from "../embeddings/generate.js";
import { config } from "../utils/config.js";
import { getPineconeIndex } from "../pinecone/client.js";

const DEFAULT_SYSTEM = `You are an analyst for Amazon-style product reviews.
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

  const queryResponse = await index.query({
    vector: queryVector,
    topK,
    includeMetadata: true,
  });

  const matches = queryResponse.matches || [];
  const blocks = matches
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
