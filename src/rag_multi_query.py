"""
RAG with Multi-Query Retrieval
==============================

Instead of embedding the user's raw question only once, the LLM first
generates N paraphrased versions of the question. Each version is used
to retrieve documents independently, and the results are de-duplicated
before being passed to the answer LLM.

This reduces the "brittleness" of retrieval: if the user's exact wording
does not match the document wording, one of the paraphrases usually will.

Usage:
    python src/rag_multi_query.py

Requires:
    - A .env file with OPENAI_API_KEY and PINECONE_API_KEY
    - A local PDF file to index
"""

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
#from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
#from langchain_community.vectorstores import FAISS
import os
from langchain_community.document_loaders import PyPDFLoader
from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
from pinecone import ServerlessSpec

load_dotenv()

# ============================================================
# 1. LLM
# ============================================================

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0
)

# ============================================================
# 2. LOAD DOCUMENT FROM FILE PATH
# ============================================================

file_path = input("Enter the path to your PDF document (.pdf): ").strip()

if not os.path.exists(file_path):
    raise FileNotFoundError(
        f"The file '{file_path}' was not found.")
if not file_path.lower().endswith(".pdf"):
    raise ValueError("Please provide a PDF file.")

loader = PyPDFLoader(file_path)
raw_documents = loader.load()
print(f"\nLoaded {len(raw_documents)} pages.")

# ============================================================
# 3. SPLIT DOCUMENTS
# ============================================================

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

chunks = text_splitter.split_documents(raw_documents)
print(f"Created {len(chunks)} chunks.")

# ============================================================
# 4. VECTOR STORE & RETRIEVER
# ============================================================

embeddings = OpenAIEmbeddings()

Pine_cone_api_key=os.getenv("PINECONE_API_KEY")
#print(Pine_cone_api_key)
pc = Pinecone(api_key=Pine_cone_api_key)

index_name = "my-rag"

existing_indexes = [index.name for index in pc.list_indexes()]

if index_name not in existing_indexes:
    pc.create_index(
        name=index_name,
        dimension=1536,
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        )
    )
# Connect to Pinecone
vector_store = PineconeVectorStore.from_documents(
    documents=chunks,
    embedding=embeddings,
    index_name=index_name
)

retriever = vector_store.as_retriever(
    search_kwargs={"k": 2}
)

# ============================================================
# 6. Multi-Query
# ============================================================

multi_query_prompt = PromptTemplate(
    template="""
You are an AI assistant helping with document retrieval.

Generate 4 different search queries based on the user's
original question.

Each query should express the same information need using
different wording or perspectives.

Do not answer the question.

Return ONLY the 4 queries, one query per line.

Original question:
{question}
""",
    input_variables=["question"]
)

# ============================================================
# 9. COMPLETE RAG CHAIN & GENERATION
# ============================================================

main_chain = (
    multi_query_prompt
    | llm
    | StrOutputParser()
)

# ============================================================
# 5. FORMAT DOCUMENTS
# ============================================================

def format_docs(docs):
    if not docs:
        return "No relevant information was retrieved."

    return "\n\n".join(
        doc.page_content
        for doc in docs
    )

# ============================================================
# 7. USER QUESTION 
# ============================================================

question = "What is RISC-V?"

parser = StrOutputParser()

# ============================================================
# 10. GENERATE MULTIPLE QUERIES
# ============================================================

print("\n" + "=" * 70)
print("GENERATING MULTIPLE QUERIES")
print("=" * 70)

generated_queries_text = main_chain.invoke(
    {
        "question": question
    }
)

# Convert the LLM output into a list
generated_queries = [
    query.strip()
    for query in generated_queries_text.split("\n")
    if query.strip()
]


print(f"\nOriginal question:")
print(question)

print("\nGenerated queries:")

for i, query in enumerate(generated_queries):
    print(f"{i + 1}. {query}")


# ============================================================
# 11. RETRIEVE DOCUMENTS FOR EACH QUERY
# ============================================================

print("\n" + "=" * 70)
print("MULTI-QUERY RETRIEVAL")
print("=" * 70)

all_retrieved_docs = []

for query in generated_queries:

    print(f"\nSearching for:")
    print(query)

    docs = retriever.invoke(query)

    print(f"Retrieved {len(docs)} documents.")

    all_retrieved_docs.extend(docs)

# ============================================================
# 12. REMOVE DUPLICATE DOCUMENTS
# ============================================================

unique_docs = {}

for doc in all_retrieved_docs:

    # Use page + content as a unique identifier
    key = (
        doc.metadata.get("page"),
        doc.page_content
    )

    unique_docs[key] = doc


retrieved_docs = list(unique_docs.values())


print("\n" + "=" * 70)
print("FINAL RETRIEVAL RESULT")
print("=" * 70)

print(
    f"Total unique documents retrieved: "
    f"{len(retrieved_docs)}"
)

# ============================================================
# 13. SHOW RETRIEVED DOCUMENTS
# ============================================================

for i, doc in enumerate(retrieved_docs):

    print("\n" + "-" * 70)
    print(f"DOCUMENT {i + 1}")
    print("-" * 70)

    print(doc.page_content)

    print("\nMetadata:")
    print(doc.metadata)

# ============================================================
# 14. FINAL RAG PROMPT
# ============================================================

answer_prompt = PromptTemplate(
    template="""
You are a helpful research assistant.

Answer the user's question ONLY using the provided context.

Do not use outside knowledge.

Do not invent information.

If the context does not contain enough information to answer
the question, say:

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
# 15. CREATE FINAL RAG CHAIN
# ============================================================

answer_chain = (
    answer_prompt
    | llm
    | StrOutputParser()
)

# ============================================================
# RETRIEVAL
# ============================================================

print("\nSearching vector database...")
#retrieved_docs = retriever.invoke(question)

# ============================================================
# 9. FORMAT MULTI-QUERY CONTEXT And Generate Final Answer
# ============================================================

context = format_docs(retrieved_docs)

response = answer_chain.invoke(
    {
        "context": context,
        "question": question
    }
)

# ============================================================
# 10. FINAL ANSWER
# ============================================================

print("\n" + "=" * 70)
print("FINAL ANSWER")
print("=" * 70)
print(response)