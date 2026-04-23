import { embedDocuments, logEmbeddingPreview } from "../embeddings/generate.js";
import { config } from "../utils/config.js";
import { getPineconeIndex } from "./client.js";

/**
 * Embed texts in batches aligned with Pinecone upsert batches.
 */
export async function embedAndUpsertReviews(vectorRecords) {
  const index = getPineconeIndex();
  const batchSize = config.pineconeUpsertBatchSize;
  const texts = vectorRecords.map((r) => r.text);
  let totalUpserted = 0;

  for (let offset = 0; offset < vectorRecords.length; offset += batchSize) {
    const slice = vectorRecords.slice(offset, offset + batchSize);
    const sliceTexts = slice.map((r) => r.text);

    const vectors = await embedDocuments(sliceTexts);
    if (vectors[0]) {
      logEmbeddingPreview(
        `batch @${offset} (first row of batch)`,
        vectors[0],
        4
      );
    }

    const upsertPayload = slice.map((row, j) => ({
      id: row.id,
      values: vectors[j],
      metadata: row.metadata,
    }));

    try {
      await index.upsert(upsertPayload);
    } catch (e) {
      const msg = e instanceof Error ? e.message : String(e);
      if (/dimension/i.test(msg)) {
        throw new Error(
          `${msg} — Set EMBEDDING_DIMENSIONS in .env to your Pinecone index dimension ` +
            `(currently ${config.embeddingDimensions} from env). ` +
            "Use text-embedding-3-small (or 3-large) so the API can emit reduced-size vectors. Restart the server after editing .env."
        );
      }
      throw e;
    }
    totalUpserted += upsertPayload.length;
    console.log(
      `[pinecone] upserted ${upsertPayload.length} vectors (total ${totalUpserted}/${vectorRecords.length})`
    );
  }

  return { upserted: totalUpserted };
}
