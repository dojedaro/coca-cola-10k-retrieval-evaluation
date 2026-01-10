📚 Coca-Cola 10-K Retrieval & Evaluation Pipeline

A production-style semantic retrieval pipeline for U.S. SEC EDGAR 10-K filings, built using LlamaIndex and open-source embeddings, with explicit retrieval evaluation and no reliance on proprietary large language models.

This project demonstrates document ingestion, chunking, vector indexing, semantic retrieval, reranking, and quantitative evaluation on real-world financial disclosures using Coca-Cola as a case study.

🎯 Project Objectives

Ingest and parse SEC EDGAR 10-K filings (2015–2024)

Convert long financial documents into searchable semantic chunks

Build vector-based retrieval using open-source embedding models

Compare standard chunk retrieval vs sentence-window retrieval

Apply cross-encoder reranking for relevance refinement

Quantitatively evaluate retrieval quality using Precision@K

🏗️ System Architecture

EDGAR Ingestion

Fetch filing metadata via the SEC Submissions API

Retrieve official 10-K HTML filings

Clean and normalize raw financial text

Chunking Strategy

Fixed-size overlapping text chunks

Metadata preserved (year, accession number, filing date)

Optional sentence-window nodes for finer-grained retrieval

Embedding & Indexing

HuggingFace sentence embeddings (MiniLM / BGE)

CPU-friendly and fully open-source

Vector index constructed with LlamaIndex

Retrieval & Reranking

Top-K semantic retrieval

Cross-encoder reranking using a MiniLM-based model

Retrieval logic runs with LLMs explicitly disabled

Evaluation

Precision@K on a manually curated financial query set

Clear exposure of both strengths and failure cases

🧠 Models & Tools Used

Framework: LlamaIndex (v0.10.x)

Embeddings:

sentence-transformers/all-MiniLM-L6-v2

BAAI/bge-small-en-v1.5

Reranker:

cross-encoder/ms-marco-MiniLM-L-6-v2

Parsing & Ingestion:

BeautifulSoup

SEC EDGAR Submissions API

Evaluation Metric:

Precision@K

🔒 LLM-based generation is intentionally excluded to ensure deterministic and auditable retrieval behavior.

📊 Example Evaluation Results
Query Type	Precision@3
Business Segments	1.00
Revenue Model	1.00
Executive Roles	0.00
Average Precision@3	0.67

These results illustrate:

Strong retrieval performance for business structure and revenue questions

Known limitations for entity-specific executive roles

The importance of explicit evaluation in retrieval systems
