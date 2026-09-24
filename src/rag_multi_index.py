"""
RAG with Multi-Index Retrieval
==============================

The same document is indexed TWICE using two different chunking
strategies:

  - SMALL index: chunk_size=500  -> fine-grained retrieval, precise
  - LARGE index: chunk_size=1200 -> coarse-grained retrieval, more context

A question is searched against both indexes and the results are
merged and de-duplicated. This combines the precision of small chunks
with the context of large chunks, without being limited to a single
chunk size.

Usage:
    python src/rag_multi_index.py

Requires:
    - A .env file with OPENAI_API_KEY and PINECONE_API_KEY
    - A local PDF file to index
"""

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore

import os
import time

load_dotenv()

# ============================================================
# 1. LLM
# ============================================================

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0
)

# ============================================================
# 2. LOAD PDF
# ============================================================

file_path = input(
    "Enter the path to your PDF document (.pdf): ").strip()

if not os.path.exists(file_path):
    raise FileNotFoundError(f"File not found: {file_path}")

if not file_path.lower().endswith(".pdf"):
    raise ValueError("Please provide a PDF file.")

loader = PyPDFLoader(file_path)
documents = loader.load()

print(f"\nLoaded {len(documents)} pages")

# ============================================================
# 3. CREATE MULTIPLE CHUNK STRATEGIES
# ============================================================

small_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)

large_splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=150)

small_chunks = small_splitter.split_documents(documents)

large_chunks = large_splitter.split_documents(documents)

print(f"Small chunks : {len(small_chunks)}")
print(f"Large chunks : {len(large_chunks)}")

# ============================================================
# 4. EMBEDDINGS
# ============================================================

embeddings = OpenAIEmbeddings()

# ============================================================
# 5. PINECONE
# ============================================================

pinecone_api_key = os.getenv("PINECONE_API_KEY")

if not pinecone_api_key:
    raise ValueError("PINECONE_API_KEY not found")

pc = Pinecone(api_key=pinecone_api_key)

# ============================================================
# 6. INDEX NAMES
# ============================================================

SMALL_INDEX = "my-rag-small"
LARGE_INDEX = "my-rag-large"

# ============================================================
# 7. CREATE INDEX IF NEEDED
# ============================================================

existing_indexes = [
    index.name
    for index in pc.list_indexes()]

for index_name in [SMALL_INDEX, LARGE_INDEX]:

    if index_name not in existing_indexes:

        print(f"Creating {index_name}")

        pc.create_index(
            name=index_name,
            dimension=1536,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )

        time.sleep(10)

# ============================================================
# 8. STORE DOCUMENTS
# ============================================================

print("\nUploading small chunks...")

small_vector_store = PineconeVectorStore.from_documents(
    documents=small_chunks,
    embedding=embeddings,
    index_name=SMALL_INDEX
)

print("Uploading large chunks...")

large_vector_store = PineconeVectorStore.from_documents(
    documents=large_chunks,
    embedding=embeddings,
    index_name=LARGE_INDEX
)

# ============================================================
# 9. RETRIEVERS
# ============================================================

small_retriever = small_vector_store.as_retriever(search_kwargs={"k": 3})

large_retriever = large_vector_store.as_retriever(search_kwargs={"k": 3})

# ============================================================
# 10. USER QUESTION
# ============================================================

question = "What is RISC-V?"

print("\n" + "=" * 70)
print("MULTI-INDEX RETRIEVAL")
print("=" * 70)

# ============================================================
# 11. SEARCH BOTH INDEXES
# ============================================================

small_docs = small_retriever.invoke(question)

large_docs = large_retriever.invoke(question)

print(f"Retrieved {len(small_docs)} docs "f"from SMALL index")

print(f"Retrieved {len(large_docs)} docs "f"from LARGE index")

# ============================================================
# 12. MERGE RESULTS
# ============================================================

all_docs = small_docs + large_docs

# ============================================================
# 13. REMOVE DUPLICATES
# ============================================================

unique_docs = {}

for doc in all_docs:

    key = (
        doc.metadata.get("page"),
        doc.page_content
    )

    unique_docs[key] = doc

retrieved_docs = list(unique_docs.values())

print(f"\nUnique docs retrieved: " f"{len(retrieved_docs)}")

# ============================================================
# 14. FORMAT DOCS
# ============================================================

def format_docs(docs):

    if not docs:
        return ("No relevant information found.")

    return "\n\n".join(
        doc.page_content
        for doc in docs
    )

context = format_docs(retrieved_docs)

# ============================================================
# 15. ANSWER PROMPT
# ============================================================

answer_prompt = PromptTemplate(
    template="""
You are a helpful research assistant.

Answer ONLY using the provided context.

If the answer is not present in the context,
say:

"I don't know."

Context:
{context}

Question:
{question}

Answer:
""",
    input_variables=[
        "context",
        "question"
    ]
)

# ============================================================
# 16. FINAL CHAIN
# ============================================================

answer_chain = (answer_prompt | llm | StrOutputParser())

# ============================================================
# 17. GENERATE ANSWER
# ============================================================

response = answer_chain.invoke(
    {
        "context": context,
        "question": question
    }
)

# ============================================================
# 18. OUTPUT
# ============================================================

print("\n" + "=" * 70)
print("FINAL ANSWER")
print("=" * 70)

print(response)