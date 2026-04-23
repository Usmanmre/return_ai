import { OpenAIEmbeddings } from "@langchain/openai";
import { config } from "../utils/config.js";

let embeddingsClient = null;
/** Recreate client if model or dimension config changes (e.g. after editing .env). */
let clientConfigKey = "";

function getClient() {
  if (!config.openaiApiKey) {
    throw new Error(
      "OPENAI_API_KEY is missing. Copy .env.example to .env and set your key."
    );
  }
  const key = `${config.openaiEmbeddingModel}:${config.embeddingDimensions}`;
  if (embeddingsClient && clientConfigKey !== key) {
    embeddingsClient = null;
  }
  if (!embeddingsClient) {
    const fields = {
      openAIApiKey: config.openaiApiKey,
      modelName: config.openaiEmbeddingModel,
    };
    /** text-embedding-3* supports explicit dimensions (must match Pinecone index). */
    if (
      config.openaiEmbeddingModel.includes("text-embedding-3") &&
      config.embeddingDimensions > 0
    ) {
      fields.dimensions = config.embeddingDimensions;
    }
    embeddingsClient = new OpenAIEmbeddings(fields);
    clientConfigKey = key;
  }
  return embeddingsClient;
}

/**
 * Embed a single string. Returns number[] from LangChain/OpenAI.
 */
export async function embedText(text) {
  const client = getClient();
  const vector = await client.embedQuery(text);
  return vector;
}

/**
 * Embed many strings (batch). Returns number[][] in the same order.
 */
export async function embedDocuments(texts) {
  const client = getClient();
  const vectors = await client.embedDocuments(texts);
  return vectors;
}

/**
 * Log embedding shape and a short preview for inspection.
 */
export function logEmbeddingPreview(label, vector, previewCount = 5) {
  const preview = vector.slice(0, previewCount).map((v) => v.toFixed(6));
  console.log(`[embeddings] ${label}`);
  console.log(`  vector length (dimensions): ${vector.length}`);
  console.log(`  first ${previewCount} values: [ ${preview.join(", ")} ]`);
}
