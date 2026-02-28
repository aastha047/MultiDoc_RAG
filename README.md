🧠 Multi-Document RAG System (Local LLM + LangChain + Chroma)

A fully local Multi-Document Retrieval-Augmented Generation (RAG) system built using:

LangChain
Chroma Vector Database
HuggingFace Embeddings
Ollama (Llama 3 – Local LLM)

This project enables users to upload multiple PDF documents (e.g., resumes) and ask grounded questions across them - including comparisons - without hallucination.

🚀 Features
📄 Multi-PDF ingestion
✂️ Smart chunking with overlap
🔢 Embedding generation using all-MiniLM-L6-v2
🗄 Persistent vector database (Chroma)
🔍 MMR-based retrieval for diversity
🤖 Local LLM (Llama3 via Ollama)
🛡 Strict anti-hallucination prompt design
💬 Interactive CLI chat interface

Architecture

User Question
      ↓
Retriever (MMR Search, k=8)
      ↓
Top Relevant Chunks from Chroma
      ↓
Prompt Template (Context-Grounded)
      ↓
Llama3 (Ollama - Local)
      ↓
Final Answer

🧠 How It Works (Step-by-Step)
Step 1: Document Loading
Loads all .pdf files from the data/ folder.
Uses PyPDFLoader to extract structured content.

Step 2: Chunking
Splits documents into:
chunk_size = 1000
chunk_overlap = 200
Preserves semantic continuity.

Step 3: Embeddings
Model: sentence-transformers/all-MiniLM-L6-v2
Converts text chunks into vector embeddings.

Step 4: Vector Storage
Uses Chroma
Persistent storage in ./chroma_db
Avoids rebuilding on every run

Step 5: Retrieval
Uses MMR (Max Marginal Relevance) for diversified retrieval

Step 6: LLM Response
Model: llama3 via Ollama
Strict prompt:
Answer ONLY using the context below.
If the answer is not present, say "I don't know."
