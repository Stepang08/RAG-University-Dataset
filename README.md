# 🎓 University RAG System

A Retrieval-Augmented Generation (RAG) pipeline that answers questions about top European universities using semantic search and Claude AI.

## What is RAG?

Large language models like Claude are trained on data up to a certain cutoff date and cannot access custom datasets. RAG solves this by:

1. **Retrieving** relevant information from your own documents at query time
2. **Augmenting** the model's prompt with that information
3. **Generating** a grounded, accurate response based on the provided context

This ensures Claude answers strictly from your data rather than relying on potentially outdated or hallucinated training knowledge.

## Project Overview

This project builds a RAG pipeline over a dataset of 332 top European universities, enabling natural language queries like:

- *"Which Swiss universities have very high research intensity?"*
- *"What are the top public universities in Germany?"*
- *"Compare academic reputation scores across Scandinavian universities"*

## Pipeline

```
CSV Dataset → Chunks → Embeddings → Normalized Vectors (stored in memory)
                                              ↑
User Query → Embedding → Cosine Similarity Search → Top 3 Chunks → Claude → Answer
```

## Tech Stack

| Component | Tool |
|---|---|
| Embeddings | `sentence-transformers` (`all-MiniLM-L6-v2`) |
| Vector storage | `numpy` (in-memory) |
| Similarity search | Dot product (after L2 normalization) |
| Generation | Anthropic Claude (`claude-opus-4-6`) |
| Environment | Google Colab |

## Dataset

`universities_base_europe.csv` — 332 European universities with the following fields:

- `ranking` — Global QS ranking
- `university_name`, `country`, `Region`
- `overall_score`, `academic_reputation`, `employer_reputation`
- `Research` — Research intensity (VH / H / MD / LO)
- `Focus`, `Size`, `Status`
- `combined_text` — Pre-formatted natural language description used as the RAG chunk

## Key Design Decisions

**One chunk = one university**: Each university's `combined_text` is self-contained (name, country, scores, research intensity). Splitting further would lose critical context needed to answer queries.

**In-memory vector store**: No external vector database (e.g. Pinecone) is used. Vectors are stored as a numpy array, making the pipeline easy to understand and modify.

**L2 normalization**: All embeddings are normalized to unit length so cosine similarity reduces to a simple dot product — computed as a single matrix multiplication over all 332 vectors.

## Setup

### 1. Install dependencies
```bash
pip install anthropic sentence-transformers numpy pandas
```

### 2. Set your API key
Store your Anthropic API key in Colab Secrets (🔑 icon in the sidebar), then load it:
```python
from google.colab import userdata
import os
os.environ["ANTHROPIC_API_KEY"] = userdata.get("ANTHROPIC_API_KEY")
```

### 3. Run the notebook
Execute cells in order. The pipeline will:
- Load and chunk the dataset
- Embed all 332 chunks using `all-MiniLM-L6-v2`
- Normalize embeddings
- Accept a natural language query and return a Claude-generated answer

## Example

**Query:** *"universities with very high research intensity in Switzerland"*

**Response:**

# Universities with Very High Research Intensity in Switzerland

Based on the provided data, all three universities listed in Switzerland have **Very High (VH)** research intensity:

| Rank | University | Global Ranking | Overall Score | Research Intensity |
|------|-----------|---------------|---------------|-------------------|
| 1 | **École Polytechnique Fédérale de Lausanne** | 22 | 90.2 | VH |
| 2 | **University of Geneva** | 155 | 61.5 | VH |
| 3 | **University of Lausanne** | 212 | 54.6 | VH |

All three are **public** institutions. Notably, École Polytechnique Fédérale de Lausanne (EPFL) stands out significantly with its global ranking of **22nd** and an overall score of **90.2**, far ahead of the other two. All three share the very high research intensity designation, reflecting Switzerland's strong commitment to research-driven higher education.

## Limitations

- Scholarship and tuition data is not included in the dataset
- Vector store is in-memory and resets on each session (no persistence)
- Limited to European universities in the QS rankings dataset

## Future Improvements

- Add a system prompt to constrain Claude to only use retrieved context
- Persist embeddings to disk to avoid re-computing on each session
- Expand dataset with program-level and scholarship information
- Add a simple chat interface with conversation history
