import dotenv from "dotenv";

dotenv.config();

export const config = {
  port: Number(process.env.PORT) || 3000,
  chunkSize: Number(process.env.CHUNK_SIZE) || 400,
  chunkOverlap: Number(process.env.CHUNK_OVERLAP) || 50,
  openaiApiKey: process.env.OPENAI_API_KEY || "",
  openaiEmbeddingModel:
    process.env.OPENAI_EMBEDDING_MODEL || "text-embedding-3-small",
  /** OpenAI chat model for RAG completion */
  openaiChatModel: process.env.OPENAI_CHAT_MODEL || "gpt-4o-mini",
  /** Pinecone serverless / pod */
  pineconeApiKey: process.env.PINECONE_API_KEY || "",
  pineconeIndexName: process.env.PINECONE_INDEX_NAME || "",
  /** Optional: serverless index host from Pinecone console (index → connect) */
  pineconeHost: process.env.PINECONE_HOST || "",
  /** Must match your Pinecone index dimension (text-embedding-3-small default = 1536) */
  embeddingDimensions: Number(process.env.EMBEDDING_DIMENSIONS) || 1536,
  /** Max vectors per Pinecone upsert batch */
  pineconeUpsertBatchSize: Number(process.env.PINECONE_UPSERT_BATCH) || 100,
  /** Metadata text truncation (Pinecone metadata size limits) */
  maxMetadataTextChars: Number(process.env.MAX_METADATA_TEXT_CHARS) || 32000,
  /** Top-k for Pinecone query + RAG context */
  ragDefaultTopK: Number(process.env.RAG_DEFAULT_TOP_K) || 8,
};
