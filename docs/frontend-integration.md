# Frontend Integration Guide

How to integrate the Pathfinder Chatbot API into your frontend application.

---

## Base URL

By default the API runs at:

```
http://localhost:8000
```

All chat-related endpoints are under the `/api` prefix.

---

## Endpoints

### 1. Send a Message — `POST /api/chat`

Send a user question and receive an AI-generated answer with sources.

**Request**

```http
POST /api/chat
Content-Type: application/json
```

| Field      | Type              | Required | Description                                           |
| ---------- | ----------------- | -------- | ----------------------------------------------------- |
| `question` | `string`          | Yes      | The user's question (1–2 000 chars).                  |
| `domain`   | `string \| null`  | No       | Filter results to a single domain (e.g. `"example.com"`). Use `domains` for multiple. |
| `domains`  | `string[] \| null`| No       | Filter results to one or more domains (e.g. `["example.com", "docs.example.com"]`). |
| `url`      | `string \| null`  | No       | Filter results to a specific URL prefix.              |
| `history`  | `array`           | No       | Previous conversation turns, oldest first (max 50).   |

> **Note:** `domain` and `domains` can be used together — they are merged into a single list. If both are provided, all specified domains are searched.

Each item in `history` has:

| Field     | Type                         | Description                  |
| --------- | ---------------------------- | ---------------------------- |
| `role`    | `"user"` \| `"assistant"`    | Who sent the message.        |
| `content` | `string`                     | The message text (1–4 000 chars). |

**Example request body (single domain)**

```json
{
  "question": "How do I enable two-factor auth?",
  "domain": "help.example.com",
  "history": [
    { "role": "user", "content": "How do I reset my password?" },
    { "role": "assistant", "content": "Go to Settings > Security and click 'Reset password'." }
  ]
}
```

**Example request body (multiple domains)**

```json
{
  "question": "How do I enable two-factor auth?",
  "domains": ["help.example.com", "docs.example.com"],
  "history": []
}
```

> **Tip:** For the first message in a conversation, omit `history` or send an empty array `[]`.

**Response `200 OK`**

```json
{
  "answer": "To reset your password, go to Settings > Security and click 'Reset password'.",
  "sources": [
    {
      "title": "Account Security — Help Center",
      "url": "https://help.example.com/security",
      "chunk_index": 2,
      "total_chunks": 5,
      "score": 0.8731
    }
  ]
}
```

| Field                  | Type     | Description                                      |
| ---------------------- | -------- | ------------------------------------------------ |
| `answer`               | `string` | The generated answer based on retrieved context.  |
| `sources`              | `array`  | List of source chunks used to build the answer.   |
| `sources[].title`      | `string` | Page title of the source.                         |
| `sources[].url`        | `string` | Original URL of the source page.                  |
| `sources[].chunk_index`| `number` | Index of this chunk within the page.              |
| `sources[].total_chunks`| `number`| Total number of chunks for that page.             |
| `sources[].score`      | `number` | Similarity score (higher = more relevant).        |

**Error responses**

| Status | Meaning                                    |
| ------ | ------------------------------------------ |
| `422`  | Validation error (e.g. empty question).    |
| `502`  | LLM or upstream service error.             |

---

### 2. List Available Domains — `GET /api/domains`

Returns all domains present in the vector store. Useful for populating a domain dropdown/filter in the UI.

**Response `200 OK`**

```json
{
  "domains": ["help.example.com", "docs.example.com"]
}
```

---

### 3. Health Check — `GET /health`

Check API and Qdrant connectivity status.

**Response `200 OK`**

```json
{
  "status": "ok",
  "provider": "gemini",
  "qdrant_connected": true
}
```

`status` is `"ok"` when everything is healthy, or `"degraded"` when Qdrant is unreachable.

---

## Frontend Examples

### Fetch (vanilla JS)

```js
async function askPathfinder(question, { domain, domains, history = [] } = {}) {
  const res = await fetch("http://localhost:8000/api/chat", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question, domain, domains, history }),
  });

  if (!res.ok) {
    throw new Error(`API error: ${res.status}`);
  }

  return res.json(); // { answer, sources }
}

// Usage — single domain
let reply = await askPathfinder("What is Pathfinder?", { domain: "example.com" });
console.log(reply.answer);

// Usage — multiple domains
reply = await askPathfinder("Compare the two products", {
  domains: ["product-a.example.com", "product-b.example.com"],
});
console.log(reply.answer);

// Usage — multi-turn conversation
const history = [];

reply = await askPathfinder("What is Pathfinder?");
console.log(reply.answer);

history.push(
  { role: "user", content: "What is Pathfinder?" },
  { role: "assistant", content: reply.answer }
);

reply = await askPathfinder("Tell me more about it", { history });
console.log(reply.answer);
```

### React (with hooks)

```tsx
import { useState, useCallback } from "react";

const API_URL = import.meta.env.VITE_API_URL ?? "http://localhost:8000";

interface ChatMessage {
  role: "user" | "assistant";
  content: string;
}

interface Source {
  title: string;
  url: string;
  chunk_index: number;
  total_chunks: number;
  score: number;
}

interface ChatResponse {
  answer: string;
  sources: Source[];
}

export function useChat() {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [history, setHistory] = useState<ChatMessage[]>([]);

  const sendMessage = useCallback(
    async (
      question: string,
      opts?: { domain?: string; domains?: string[] }
    ): Promise<ChatResponse | null> => {
      setLoading(true);
      setError(null);

      try {
        const res = await fetch(`${API_URL}/api/chat`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            question,
            domain: opts?.domain ?? null,
            domains: opts?.domains ?? null,
            history,
          }),
        });

        if (!res.ok) {
          throw new Error(`Server responded with ${res.status}`);
        }

        const data = (await res.json()) as ChatResponse;

        // Append this exchange to history for subsequent turns
        setHistory((prev) => [
          ...prev,
          { role: "user", content: question },
          { role: "assistant", content: data.answer },
        ]);

        return data;
      } catch (err: any) {
        setError(err.message);
        return null;
      } finally {
        setLoading(false);
      }
    },
    [history]
  );

  const clearHistory = useCallback(() => setHistory([]), []);

  return { sendMessage, clearHistory, history, loading, error };
}
```

```tsx
function ChatBox() {
  const { sendMessage, clearHistory, loading, error } = useChat();
  const [answer, setAnswer] = useState("");
  const [sources, setSources] = useState([]);

  async function handleSubmit(question: string) {
    const data = await sendMessage(question);
    if (data) {
      setAnswer(data.answer);
      setSources(data.sources);
    }
  }

  return (
    <div>
      <button onClick={clearHistory}>New conversation</button>
      {loading && <p>Thinking…</p>}
      {error && <p className="error">{error}</p>}
      {answer && <p>{answer}</p>}
      {sources.map((s, i) => (
        <a key={i} href={s.url} target="_blank" rel="noopener noreferrer">
          {s.title}
        </a>
      ))}
    </div>
  );
}
```

### Fetching Domains for a Filter

```ts
async function fetchDomains(): Promise<string[]> {
  const res = await fetch(`${API_URL}/api/domains`);
  const { domains } = await res.json();
  return domains;
}
```

---

## CORS

The API ships with a permissive CORS policy (`allow_origins=["*"]`), so requests from any frontend origin are accepted in development. For production, update the allowed origins in the FastAPI middleware configuration.

---

## Interactive API Docs

FastAPI auto-generates interactive documentation:

| Docs     | URL                              |
| -------- | -------------------------------- |
| Swagger  | `http://localhost:8000/docs`     |
| ReDoc    | `http://localhost:8000/redoc`    |

Use these to test endpoints directly from the browser.
