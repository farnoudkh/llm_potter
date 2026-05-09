# Harry Potter RAG Chatbot

A conversational Q&A chatbot that answers questions about the Harry Potter book series using **Retrieval-Augmented Generation (RAG)**. The model is strictly grounded — it only uses passages retrieved from the books and never relies on its training memory.

## How it works

```
User question
      │
      ├──► 1. HyDE: Claude generates a hypothetical book passage matching the question
      │
      ├──► 2. FAISS searches the book index with both the question and the HyDE passage
      │         → returns the most semantically relevant chunks
      │
      └──► 3. Retrieved passages are injected into Claude's context
                → Claude answers strictly from those passages
```

**Key techniques:**

- **RAG (Retrieval-Augmented Generation)** — separates knowledge storage (FAISS index) from reasoning (Claude), so the model can only answer from the actual books
- **HyDE (Hypothetical Document Embedding)** — instead of searching with the raw question, Claude first writes a fake book-style passage that would answer it; this passage embeds much closer to real book chunks, dramatically improving retrieval on indirect or paraphrased questions
- **Strict grounding** — the system prompt forbids Claude from using its training knowledge; if the answer isn't in the retrieved passages, it says so explicitly
- **Prompt caching** — the system prompt uses Anthropic's `cache_control` to reduce API costs on repeated calls

## Stack

| Layer | Technology |
|---|---|
| Embedding & retrieval | HuggingFace `paraphrase-multilingual-MiniLM-L12-v2` + FAISS |
| Language model | Anthropic Claude Haiku (`claude-haiku-4-5-20251001`) |
| UI | Streamlit |
| Vector store | FAISS (pre-built index) |

## Setup

### 1. Clone and create a virtual environment

```bash
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure your API key

```bash
cp .env.example .env
```

Edit `.env` and add your Anthropic key:

```
ANTHROPIC_API_KEY=sk-ant-...
```

Get your key at [console.anthropic.com](https://console.anthropic.com).

### 4. Run

```bash
streamlit run script.py
```

The app opens at `http://localhost:8501`.

## Project structure

```
.
├── script.py                  # Main app (RAG pipeline + Streamlit UI)
├── rebuild_index.py           # One-time script to rebuild the FAISS index
├── harry_potter_hf_index/     # Pre-built FAISS vector index (HuggingFace embeddings)
├── requirements.txt
├── .env.example
└── .gitignore
```

## Notes

- The FAISS index is pre-built and must be present in the project root. Run `rebuild_index.py` to regenerate it from a source index.
- Embeddings run locally via HuggingFace — no OpenAI key required.
- Conversation history is maintained within the session and resets on page refresh.
