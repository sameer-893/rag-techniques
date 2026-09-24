# Multi-Index Retrieval — Deep Dive

## Why

Chunk size is a hard compromise:

- **Small chunks (500 chars)** — retrieval is precise, but a single chunk may
  lack the surrounding context needed to answer correctly.
- **Large chunks (1200 chars)** — rich context, but embedding averages dilute
  the specific fact, hurting precision.

Multi-index retrieval **indexes the document twice** and queries both,
getting the best of both worlds.

## Flow

1. The PDF is split twice: `chunk_size=500, overlap=50` and
   `chunk_size=1200, overlap=150`.
2. Each chunk set is embedded into its own Pinecone index
   (`rag-multi-index-small`, `rag-multi-index-large`).
3. The user question is searched against **both** indexes
   (`TOP_K = 3` per index).
4. Results are merged and de-duplicated using
   `(page, page_content)` as the key.
5. The union becomes the context for the final answer.

## Trade-offs

| Benefit | Cost |
|---|---|
| Combines precision + context | 2× embedding & storage cost |
| No extra LLM call per question | More indexes to manage |
| Works well for mixed document types | Duplicate content across indexes |

## Tuning tips

- Adjust chunk sizes per document type (code/docs vs. prose).
- Try `TOP_K` asymmetrically, e.g. 2 small + 3 large, if one index proves
  more useful for your data.
- For many chunk sizes, consider LangChain's
  [ParentDocumentRetriever](https://python.langchain.com/docs/how_to/parent_document_retriever)
  which embeds small chunks but returns their larger parents.
