# Coca-Cola 10-K Retrieval and Evaluation

**Semantic search over ten years of Coca-Cola's SEC 10-K filings (2015–2024) with open-source embeddings and a cross-encoder reranker, and no LLM in the retrieval path, so every result is deterministic and auditable.**

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dojedaro/coca-cola-10k-retrieval-evaluation/blob/main/coca_cola_10k_retrieval_pipeline.ipynb)
[![Notebook checks](https://github.com/dojedaro/coca-cola-10k-retrieval-evaluation/actions/workflows/notebooks.yml/badge.svg)](https://github.com/dojedaro/coca-cola-10k-retrieval-evaluation/actions/workflows/notebooks.yml)

> Part of a series of RAG projects: [strict PDF QA](https://github.com/dojedaro/strict-pdf-question-answering) ·
> [PDF RAG + ReAct agent](https://github.com/dojedaro/pdf-rag-react-agent-langchain) ·
> [policy QA with evaluation](https://github.com/dojedaro/openai-rag-policy-qa-evaluation) ·
> **10-K retrieval evaluation** ·
> [research-paper RAG](https://github.com/dojedaro/research-paper-rag-qa-system)

## Results

10 annual filings were fetched from SEC EDGAR, cleaned and split into **1,153 chunks**
(2,000 characters, 200 overlap). Retrieval uses `bge-small-en-v1.5` embeddings, top 4.

| Query | Precision@3 |
|---|---|
| What are Coca-Cola's main business segments? | 1.00 |
| How does Coca-Cola make money? | 1.00 |
| Who is Coca-Cola's Chief Information Officer? | **0.00** |
| **Average** | **0.67** |

**What this does and doesn't show.** Three queries is a smoke test, not a benchmark. A chunk counts
as relevant if it contains the query's expected keywords, which is a rough proxy for relevance. And
the scores above come from the embedding retriever alone; the cross-encoder reranker is demonstrated
in the notebook but wasn't scored. The useful finding is the failure: **a question about one named
person scores 0**. Dense embeddings capture topics ("business segments") far better than specific
entities and titles, which is the classic case for adding keyword (BM25) search alongside them.

## How it works

```mermaid
flowchart LR
    E[SEC EDGAR<br/>Submissions API] --> H[10-K HTML<br/>2015-2024]
    H --> C[Clean text, chunk<br/>2,000 chars, 200 overlap<br/>+ year and accession metadata]
    C --> I[Vector index<br/>LlamaIndex + bge-small]
    C --> W[Sentence-window index<br/>window of 3 sentences]
    I --> R[Top-4 retrieval]
    R --> X[Cross-encoder rerank<br/>ms-marco-MiniLM-L-6]
    R --> P[Precision@3<br/>keyword relevance]
```

- **Ingestion:** filing metadata from the SEC Submissions API, then the official 10-K HTML, cleaned
  with BeautifulSoup. Each chunk keeps its fiscal year, accession number and filing date.
- **Two index types:** fixed-size chunks, and sentence-window nodes that retrieve one sentence but
  return its neighbours as context.
- **Retrieval only:** LlamaIndex runs with `llm=None`, so nothing is generated and every answer is a
  quoted passage.

## Run it

```bash
pip install -r requirements.txt
jupyter notebook coca_cola_10k_retrieval_pipeline.ipynb
```

SEC EDGAR requires a `User-Agent` header that identifies you (name and email). Set it in the
ingestion cell before running. No API keys are needed; the models run on CPU.

## Stack

Python, LlamaIndex, sentence-transformers (all-MiniLM-L6-v2, bge-small-en-v1.5), cross-encoder
reranking, BeautifulSoup, SEC EDGAR API.

## Limitations and next steps

- Build a labeled set of 30+ questions with known answer passages, and report Precision@k, Recall@k
  and MRR for each configuration: chunks vs. sentence windows, with and without the reranker.
- Add BM25 and hybrid retrieval, which should fix the entity questions dense search misses.
- Filter by fiscal year from the metadata, so "2019 revenue" can't retrieve a 2023 passage.

## License

MIT
