# Pathfinder Chatbot

A **RAG (Retrieval-Augmented Generation) chatbot API** that answers questions using crawled website content stored in [Qdrant](https://qdrant.tech/). Built with FastAPI, LangChain, and supports both Google Gemini and self-hosted LLMs.

Pathfinder retrieves relevant text chunks from a Qdrant vector store, injects them as context into a prompt, and generates grounded answers — citing the source page title and URL. It pairs with the [crawl-data-for-ai](https://github.com/skill-wanderer/crawl-data-for-ai) pipeline that populates the vector store.

---

## Features

- **Semantic search** over website content via Qdrant vector similarity
- **Domain & URL filtering** — scope answers to a specific website or URL prefix
- **Provider-agnostic** — swap between Google Gemini and any OpenAI-compatible self-hosted model (Ollama, vLLM, etc.) with a single env var
- **Health endpoint** with Qdrant connectivity check
- **Docker Compose** setup for one-command deployment (API + Qdrant)

---

## Architecture

```
┌────────────┐     POST /api/chat      ┌──────────────────┐
│   Client   │ ──────────────────────▶  │   FastAPI App    │
└────────────┘                          │                  │
                                        │  1. Embed query  │
                                        │  2. Search Qdrant│
                                        │  3. Build prompt │
                                        │  4. Call LLM     │
                                        └────────┬─────────┘
                                                 │
                              ┌───────────────────┼───────────────────┐
                              ▼                                       ▼
                     ┌────────────────┐                     ┌─────────────────┐
                     │     Qdrant     │                     │   LLM Provider  │
                     │  Vector Store  │                     │ Gemini / Ollama │
                     └────────────────┘                     └─────────────────┘
```

---

## Quick Start

### Prerequisites

- **Python 3.12+**
- A running **Qdrant** instance (or use Docker Compose below)
- A **Google Gemini API key** — or a self-hosted model endpoint

### 1. Clone & install

```bash
git clone https://github.com/skill-wanderer/Pathfinder-chatbot.git
cd Pathfinder-chatbot

python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

### 2. Configure environment

Create a `.env` file in the project root:

```env
# --- LLM provider: "gemini" or "selfhost" ---
LLM_PROVIDER=gemini

# --- Google Gemini (when LLM_PROVIDER=gemini) ---
GEMINI_API_KEY=your-gemini-api-key
GEMINI_LLM_MODEL=gemini-2.0-flash
GEMINI_EMBEDDING_MODEL=gemini-embedding-001

# --- Self-hosted / Ollama (when LLM_PROVIDER=selfhost) ---
# SELFHOST_BASE_URL=http://localhost:11434/v1
# SELFHOST_API_KEY=not-needed
# SELFHOST_LLM_MODEL=llama3
# SELFHOST_EMBEDDING_MODEL=nomic-embed-text
# SELFHOST_EMBEDDING_DIMENSIONS=3072

# --- Qdrant ---
QDRANT_HOST=localhost
QDRANT_PORT=6333
QDRANT_COLLECTION=website_pages

# --- RAG tuning ---
RAG_TOP_K=5
RAG_SCORE_THRESHOLD=0.5
```

### 3. Run

```bash
uvicorn app.main:app --reload
```

The API is now available at **http://localhost:8000**. Interactive docs at [http://localhost:8000/docs](http://localhost:8000/docs).

---

## Docker Compose

Spin up both the API and Qdrant with a single command:

```bash
docker compose up --build
```

This starts:

| Service        | Port  | Description              |
| -------------- | ----- | ------------------------ |
| `pathfinder`   | 8000  | Chatbot API              |
| `qdrant`       | 6333  | Qdrant REST API          |
| `qdrant`       | 6334  | Qdrant gRPC              |

Qdrant data is persisted in a named Docker volume (`qdrant_data`).

---

## API Endpoints

### `POST /api/chat`

Ask a question. Returns an LLM-generated answer with source citations.

**Request body:**

```json
{
  "question": "What services does example.com offer?",
  "domain": "example.com",
  "url": null
}
```

| Field      | Type     | Required | Description                              |
| ---------- | -------- | -------- | ---------------------------------------- |
| `question` | `string` | Yes      | The user's question (1 – 2000 chars)     |
| `domain`   | `string` | No       | Filter results to a specific domain      |
| `url`      | `string` | No       | Filter results to a specific URL prefix  |

**Response:**

```json
{
  "answer": "Example.com offers web development, SEO, and cloud hosting services.",
  "sources": [
    {
      "title": "Our Services",
      "url": "https://example.com/services",
      "chunk_index": 0,
      "total_chunks": 3,
      "score": 0.8721
    }
  ]
}
```

### `GET /api/domains`

List all domains available in the vector store.

**Response:**

```json
{
  "domains": ["example.com", "docs.example.com"]
}
```

### `GET /health`

Health check with Qdrant connectivity status.

**Response:**

```json
{
  "status": "ok",
  "provider": "gemini",
  "qdrant_connected": true
}
```

---

## Project Structure

```
├── app/
│   ├── main.py              # FastAPI app, lifespan, CORS, health endpoint
│   ├── config.py            # Pydantic Settings (env vars)
│   ├── dependencies.py      # Singleton factories (Qdrant, embeddings, LLM)
│   ├── models/
│   │   └── schemas.py       # Request / response Pydantic models
│   ├── routers/
│   │   └── chat.py          # /api/chat and /api/domains routes
│   └── services/
│       ├── embeddings.py    # Embedding provider abstraction
│       ├── llm.py           # LLM provider abstraction
│       └── retriever.py     # Qdrant search & domain listing
├── docker-compose.yml
├── Dockerfile
├── requirements.txt
├── qdrant-vector-schema.md  # Schema docs for the Qdrant collection
└── .env                     # Your local config (not committed)
```

---

## Qdrant Vector Schema

The vector store is populated by the companion [crawl-data-for-ai](https://github.com/skill-wanderer/crawl-data-for-ai) pipeline. Each point represents a text chunk from a crawled page:

| Payload Field      | Type       | Description                                    |
| ------------------ | ---------- | ---------------------------------------------- |
| `text`             | `string`   | The raw text chunk fed to the LLM as context   |
| `url`              | `string`   | Full URL of the source page                    |
| `domain`           | `string`   | Root domain (e.g. `example.com`)               |
| `title`            | `string`   | HTML `<title>` of the page                     |
| `meta_description` | `string`   | Meta description tag content                   |
| `headings`         | `string[]` | Extracted h1/h2/h3 headings                    |
| `chunk_index`      | `int`      | Zero-based chunk index within the page         |
| `total_chunks`     | `int`      | Total chunks the page was split into           |

Vectors are **3072-dimensional** (Cosine distance), generated by `gemini-embedding-001`. See [qdrant-vector-schema.md](qdrant-vector-schema.md) for full details.

---

## License

[Apache License 2.0](LICENSE)