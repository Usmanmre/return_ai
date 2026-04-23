import { Pinecone } from "@pinecone-database/pinecone";
import { config } from "../utils/config.js";

/**
 * Return a Pinecone index handle for upsert/query.
 * Set PINECONE_HOST if your Pinecone console shows a host URL for the index.
 */
export function getPineconeIndex() {
  if (!config.pineconeApiKey) {
    throw new Error("PINECONE_API_KEY is missing in .env");
  }
  if (!config.pineconeIndexName) {
    throw new Error("PINECONE_INDEX_NAME is missing in .env");
  }

  const pc = new Pinecone({ apiKey: config.pineconeApiKey });
  if (config.pineconeHost) {
    return pc.index(config.pineconeIndexName, config.pineconeHost);
  }
  return pc.index(config.pineconeIndexName);
}
