/**
 * In-memory vector store: a plain array of records.
 * No LangChain VectorStore — just explicit state you can inspect.
 */

/** @type {{ id: string, text: string, embedding: number[], metadata: Record<string, unknown> }[]} */
let records = [];

export function clearStore() {
  records = [];
}

export function addRecords(newRecords) {
  records.push(...newRecords);
}

export function getAllRecords() {
  return records;
}

export function count() {
  return records.length;
}
