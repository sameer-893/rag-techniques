# RAG Techniques: Multi-Query & Multi-Index Retrieval

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![LangChain](https://img.shields.io/badge/LangChain-%E2%9D%A4%EF%B8%8F-green)](https://www.langchain.com/)
[![Pinecone](https://img.shields.io/badge/Pinecone-Vector%20DB-000000?logo=pinecone)](https://www.pinecone.io/)
[![OpenAI](https://img.shields.io/badge/OpenAI-gpt--4o--mini-412991?logo=openai)](https://platform.openai.com/)

Two practical implementations of **Retrieval-Augmented Generation (RAG)** that attack the same problem from different angles: **poor retrieval quality**. Both scripts load a PDF, embed it into a [Pinecone](https://www.pinecone.io/) vector database, and answer user questions using [LangChain](https://www.langchain.com/) + OpenAI.

## The Problem with Naive RAG

In the simplest RAG pipeline, the user's question is embedded once and matched against document chunks. This is fragile:

- **Wording mismatch** — if the user asks *"What is RISC-V?"* but the document says *"RISC-V is an open-standard instruction set architecture..."*, keyword-style or embedding gaps can cause misses.
- **One-size-fits-all chunking** — small chunks (500 chars) are precise but lack context; large chunks (1200 chars) have context but are imprecise.

## The Two Techniques

| | Multi-Query | Multi-Index |
|---|---|---|
| **File** | [`src/rag_multi_query.py`](src/rag_multi_query.py) | [`src/rag_multi_index.py`](src/rag_multi_index.py) |
| **Idea** | Paraphrase the question N times, search with all versions | Index the same document with 2 chunk sizes, search both |
| **What improves** | Recall — different phrasings match different document wording | Context — small chunks find facts, large chunks supply context |
| **Cost** | N× retrieval calls + 1 LLM call to generate queries | 2× storage & embedding cost |
| **Best for** | Questions whose wording may differ from the document | Documents needing both precision and broad context |

### How Multi-Query works

```
User question
     │
     ▼
LLM generates 4 paraphrased queries
     │
     ▼
Retrieve top-k docs for EACH query ──► de-duplicate ──► merged context
                                                          │
                                                          ▼
                                              LLM answers with context
```

### How Multi-Index works

```
PDF ──► small chunks (500)  ──► SMALL index ─┐
    └──► large chunks (1200) ──► LARGE index ─┤
                                              ▼
                              question searched in BOTH ──► merge & de-dup
                                                                  │
                                                                  ▼
                                                        LLM answers with context
```

## Project Structure

```
rag-techniques/
├── README.md                  # you are here
├── requirements.txt
├── .env.example               # copy to .env and add your keys
├── src/
│   ├── rag_multi_query.py     # technique 1
│   └── rag_multi_index.py     # technique 2
├── docs/
│   ├── multi_query.md         # deep dive: technique 1
│   └── multi_index.md         # deep dive: technique 2
└── data/                      # put your PDFs here (git-ignored)
```

## Setup

### 1. Clone & install

```bash
git clone https://github.com/<your-username>/rag-techniques.git
cd rag-techniques

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

### 2. Configure API keys

```bash
cp .env.example .env
```

Edit `.env` and add your keys:

```
OPENAI_API_KEY=sk-...
PINECONE_API_KEY=...
```

> ⚠️ **Never commit your `.env` file.** It is already listed in `.gitignore`.

### 3. Run

```bash
# Technique 1: Multi-Query
python src/rag_multi_query.py

# Technique 2: Multi-Index
python src/rag_multi_index.py
```

Each script will:
1. Ask for the path to a PDF (e.g. `data/my_document.pdf`)
2. Chunk, embed, and upload it to Pinecone (first run only)
3. Ask your question
4. Print the grounded answer

## Configuration

Both scripts have a `Configuration` section at the top:

| Setting | Default | Description |
|---|---|---|
| `LLM_MODEL` | `gpt-4o-mini` | Chat model used for queries & answers |
| `TOP_K` | `2` / `3` | Documents retrieved per query / index |
| `NUM_QUERIES` | `4` | Paraphrased queries (multi-query only) |
| `EMBEDDING_DIMENSION` | `1536` | Must match `text-embedding-3-small` |

## Notes & Limitations

- **Re-indexing**: Both scripts upload the PDF on every run. To avoid duplicate uploads, delete the Pinecone indexes between runs, or check `pc.describe_index(...)` stats before uploading.
- **Costs**: Each run makes OpenAI embedding + chat calls, plus Pinecone storage. Multi-query multiplies retrieval calls by `NUM_QUERIES`.
- **Production alternative**: LangChain ships a built-in [`MultiQueryRetriever`](https://python.langchain.com/docs/how_to/MultiQueryRetriever) — these scripts implement the logic manually for educational clarity.

## License

MIT — see [LICENSE](LICENSE).
