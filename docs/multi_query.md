# Multi-Query Retrieval — Deep Dive

## Why

A single embedding of the user's question is sensitive to phrasing. Documents
often answer a question with different words than the user used. Multi-query
retrieval compensates by **generating paraphrases** of the question and
retrieving against each one.

## Flow

1. The question is sent to the LLM with a prompt asking for
   `NUM_QUERIES` (default 4) rephrased versions.
2. Each generated query is run independently against the vector store
   (`TOP_K = 2` docs per query).
3. All retrieved documents are merged and **de-duplicated** using
   `(page, page_content)` as the unique key.
4. The de-duplicated union becomes the context for the final answer.

## Trade-offs

| Benefit | Cost |
|---|---|
| Higher recall on paraphrased questions | `NUM_QUERIES` × retrieval calls |
| Robust to wording mismatch | One extra LLM call per question |
| Simple to implement & tune | May pull in loosely related chunks |

## Tuning tips

- Increase `NUM_QUERIES` for short/vague questions; decrease for precise ones.
- Lower `TOP_K` when documents are long to keep context within the LLM window.
- If de-duplication still leaves too many chunks, add a re-ranking step
  (e.g., Cohere Rerank) after merging.

## Reference

- LangChain built-in: https://python.langchain.com/docs/how_to/MultiQueryRetriever
