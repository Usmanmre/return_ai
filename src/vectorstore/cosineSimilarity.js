/**
 * Cosine similarity for two vectors of equal length.
 *
 * Math (for vectors a and b in R^n):
 *   dot(a, b)     = sum_i( a_i * b_i )
 *   ||a||         = sqrt( sum_i( a_i^2 ) )   (Euclidean length)
 *   cos(theta)    = dot(a, b) / (||a|| * ||b||)
 *
 * Interpretation:
 *   - cos(theta) is the cosine of the angle between a and b when both are drawn from the origin.
 *   - Range is [-1, 1] for arbitrary real vectors; embedding vectors from the same model
 *     often behave like narrow cones, so practical scores for "related" text tend to cluster high.
 *   - 1 means same direction, 0 means orthogonal, -1 means opposite direction.
 */
export function cosineSimilarity(a, b) {
  if (a.length !== b.length) {
    throw new Error(
      `Vector length mismatch: ${a.length} vs ${b.length} (same embedding model required)`
    );
  }
  let dot = 0;
  let normA = 0;
  let normB = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
    normA += a[i] * a[i];
    normB += b[i] * b[i];
  }
  const denom = Math.sqrt(normA) * Math.sqrt(normB);
  if (denom === 0) return 0;
  return dot / denom;
}
