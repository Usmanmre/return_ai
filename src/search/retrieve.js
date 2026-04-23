import { embedText, logEmbeddingPreview } from "../embeddings/generate.js";
import { getAllRecords } from "../vectorstore/memoryStore.js";
import { cosineSimilarity } from "../vectorstore/cosineSimilarity.js";

/**
 * Embed the query, score every stored vector with cosine similarity, return top-k.
 */
export async function searchTopK(query, k = 5) {
  const queryVec = await embedText(query);
  logEmbeddingPreview("query embedding", queryVec);

  const rows = getAllRecords();
  const scored = rows.map((row) => {
    const score = cosineSimilarity(queryVec, row.embedding);
    return { ...row, score };
  });

  scored.sort((x, y) => y.score - x.score);
  const top = scored.slice(0, Math.max(0, Math.min(k, scored.length)));

  console.log("[search] similarity scores (all results, sorted desc):");
  for (const r of scored) {
    console.log(`  id=${r.id} score=${r.score.toFixed(6)}`);
  }

  return top.map(({ id, text, metadata, score }) => ({
    id,
    text,
    metadata,
    score,
  }));
}
