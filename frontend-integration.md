# Pathfinder — Frontend Integration & Design Guide

A complete reference for frontend developers to integrate the Pathfinder chatbot **and** match its visual design. See `DESIGN_SYSTEM.md` for the broader Skill-Wanderer design tokens that Pathfinder inherits from.

---

## Table of Contents

1. [Design Philosophy](#1-design-philosophy)
2. [Color Palette](#2-color-palette)
3. [Component Anatomy](#3-component-anatomy)
4. [Floating Action Button (FAB)](#4-floating-action-button-fab)
5. [Chat Panel](#5-chat-panel)
6. [Panel Header](#6-panel-header)
7. [Welcome Screen](#7-welcome-screen)
8. [Message Bubbles](#8-message-bubbles)
9. [Sources Section](#9-sources-section)
10. [Input Bar](#10-input-bar)
11. [Typing Indicator](#11-typing-indicator)
12. [Error State](#12-error-state)
13. [Animations & Motion](#13-animations--motion)
14. [Responsive / Mobile](#14-responsive--mobile)
15. [Typography](#15-typography)
16. [Composable API — `usePathfinder`](#16-composable-api--usepathfinder)
17. [Runtime Config](#17-runtime-config)
18. [Chat API Reference](#18-chat-api-reference)
19. [CSS Class Reference (quick-look)](#19-css-class-reference)

---

## 1. Design Philosophy

Pathfinder uses a **"Space Explorer"** visual metaphor layered on top of the core Skill-Wanderer dark-first brand:

| Attribute | Value |
|---|---|
| **Theme** | Deep-space dark, warm orange/gold accents |
| **Metaphor** | Planet exploration — compass FAB, starfield background, rocket send button, "expedition" resets |
| **Mood** | Friendly, adventurous, lightweight |
| **Panel style** | Floating card, bottom-right, glass-dark surface |

Everything should feel like a small command console orbiting the page, not a traditional chat widget.

---

## 2. Color Palette

Pathfinder uses a focused subset of the brand tokens **plus** a few component-specific values.

### Brand tokens (inherited from `main.css` / `DESIGN_SYSTEM.md`)

| Token | Hex | Usage |
|---|---|---|
| `--primary-orange` | `#FF6B35` | Gradient start, borders, FAB, send button |
| `--deep-orange` | `#E85D25` | Gradient end, user bubble end |
| `--accent-yellow` | `#FFD93D` | Title gradient end, source links, hover accents |
| `--light-text` | `#e0e0e0` | General body text |

### Component-specific values (hard-coded in `PathfinderChat.vue`)

| Name | Value | Where |
|---|---|---|
| Panel background | `#0d1117` | `.pathfinder-panel` main bg |
| Mid orange | `#FF8C42` | Compass icon glow, planet icon, typing dots, avatar icon |
| Starfield dots | `#fff` (pure white, faded via opacity) | `.pathfinder-star` |
| Assistant bubble bg | `rgba(255,255,255,0.06)` | `.pathfinder-msg-assistant .pathfinder-bubble` |
| Assistant bubble border | `rgba(255,255,255,0.06)` | Same |
| User bubble gradient | `linear-gradient(135deg, #FF6B35, #E85D25)` | `.pathfinder-msg-user .pathfinder-bubble` |
| FAB border | `rgba(255, 217, 61, 0.3)` | `.pathfinder-fab` |
| Panel border | `rgba(255, 217, 61, 0.12)` | `.pathfinder-panel` |
| Header gradient | `linear-gradient(135deg, rgba(255,107,53,0.12) 0%, rgba(13,17,23,0.95) 100%)` | `.pathfinder-header` |
| Error text | `#ff6b6b` | `.pathfinder-error` |
| Suggestion pill bg | `rgba(255,107,53,0.08)` | `.pathfinder-suggestion` |
| Suggestion pill border | `rgba(255,107,53,0.2)` | `.pathfinder-suggestion` |
| Suggestion pill text | `#FF8C42` | `.pathfinder-suggestion` |

---

## 3. Component Anatomy

```
┌──────────────────────────────────────────────┐
│  Page content                                │
│                                              │
│                      ┌────────────────────┐  │
│                      │ ◈ Pathfinder  ⟲ ✕ │  │   ← Header
│                      ├────────────────────┤  │
│                      │ ✦  ✦      ✦       │  │   ← Starfield (decorative)
│                      │                    │  │
│                      │  [Welcome / Msgs]  │  │   ← Messages area
│                      │                    │  │
│                      │  Discovered Sources│  │   ← Sources (after response)
│                      ├────────────────────┤  │
│                      │ [Where shall we…] 🚀│  │   ← Input bar
│                      └────────────────────┘  │
│                                        ⊕     │   ← FAB (visible when panel closed)
└──────────────────────────────────────────────┘
```

**Files involved:**

| File | Role |
|---|---|
| `components/PathfinderChat.vue` | Full template + styles (single-file component) |
| `composables/usePathfinder.ts` | State management, API calls, session persistence |
| `nuxt.config.ts` → `runtimeConfig.public.pathfinder` | API URL, domains, session settings |

---

## 4. Floating Action Button (FAB)

The FAB is visible when the panel is **closed**. It is a compass icon.

| Property | Value |
|---|---|
| Position | `fixed; bottom: 1.5rem; right: 1.5rem` |
| Size | `60px × 60px` |
| Shape | Circle (`border-radius: 50%`) |
| Background | `radial-gradient(circle at 35% 35%, #FF8C42, #FF6B35 50%, #E85D25)` |
| Border | `2px solid rgba(255, 217, 61, 0.3)` |
| Shadow | `0 4px 24px rgba(255,107,53,0.45)` |
| Icon | Compass SVG, `28×28`, white stroke |
| z-index | `9998` |
| Animation | Slow pulse (`pathfinder-pulse`, 3 s, pauses on hover) |
| Outer ring | Dashed orbit ring (`1.5px dashed rgba(255,217,61,0.25)`), slowly spins 360° over 12 s |
| Hover | `scale(1.1)`, stronger shadow, pulse stops |

```css
/* Key FAB styles */
.pathfinder-fab {
  width: 60px;
  height: 60px;
  border-radius: 50%;
  background: radial-gradient(circle at 35% 35%, #FF8C42, #FF6B35 50%, #E85D25);
  border: 2px solid rgba(255, 217, 61, 0.3);
  box-shadow: 0 4px 24px rgba(255, 107, 53, 0.45);
  animation: pathfinder-pulse 3s ease-in-out infinite;
}
.pathfinder-fab:hover {
  transform: scale(1.1);
  animation: none;
}
```

---

## 5. Chat Panel

Appears in place of the FAB when opened.

| Property | Value |
|---|---|
| Position | `fixed; bottom: 1.5rem; right: 1.5rem` |
| Size | `390px` wide × `540px` tall |
| Max size | `calc(100vw - 2rem)` wide, `calc(100vh - 3rem)` tall |
| Background | `#0d1117` |
| Border | `1px solid rgba(255, 217, 61, 0.12)` |
| Border radius | `18px` |
| Shadow | `0 8px 48px rgba(0,0,0,0.6), 0 0 1px rgba(255,107,53,0.3), inset 0 1px 0 rgba(255,255,255,0.04)` |
| z-index | `9999` |
| Layout | Flexbox column: Header → Messages (flex:1, scrollable) → Input bar |
| Transition (open/close) | Slide-up + fade + slight scale (`0.3s ease`) |

### Starfield overlay

- 20 visible stars (`:nth-child(n+21)` are hidden)
- Each star: `2px` white circle, fades in/out via `pathfinder-twinkle` (4 s)
- Odd children: 3 s duration; every 3rd child: 5 s, 1 px size
- Positioned absolutely across the panel; `pointer-events: none`

---

## 6. Panel Header

| Property | Value |
|---|---|
| Height | Auto (~44 px with padding) |
| Padding | `0.75rem 1rem` |
| Background | `linear-gradient(135deg, rgba(255,107,53,0.12) 0%, rgba(13,17,23,0.95) 100%)` |
| Bottom border | `1px solid rgba(255, 217, 61, 0.08)` |
| Layout | Flex row, `space-between` |

**Left side:**
- Planet SVG icon (`22×22`, color `#FF8C42`, gentle bob animation `pathfinder-orbit-bob` 6 s)
- Title "**Pathfinder**" — gradient text (`linear-gradient(135deg, #FFD93D, #FF6B35)` + `background-clip: text`)
- Subtitle "EXPLORATION GUIDE" — `10px`, 50% opacity, wide tracking

**Right side:**
- **Reset button** (refresh icon) — clears conversation, tooltip "New expedition"
- **Close button** (× icon)
- Both: transparent bg, `rgba(255,255,255,0.45)` icon color; on hover → yellow `#FFD93D` + subtle bg tint

---

## 7. Welcome Screen

Shown when `history` is empty (no messages yet).

| Element | Details |
|---|---|
| Container | Centered text, `padding: 1.5rem 1rem 1rem`, color `#b0b0b0` |
| Planet icon | `48×48` SVG (planet + orbit ring + small stars), color `#FF8C42`, bob animation |
| Title | "Welcome, Explorer!" — `text-sm font-semibold` |
| Description | "Chart your course through Skill-Wanderer's universe…" — `text-xs`, 60% opacity |
| Suggestion pills | Flex-wrap row, gap `6px`, centered |

### Suggestion pills

| Property | Value |
|---|---|
| Background | `rgba(255,107,53,0.08)` |
| Border | `1px solid rgba(255,107,53,0.2)` |
| Text color | `#FF8C42` |
| Font size | `0.6875rem` (11 px) |
| Padding | `5px 10px` |
| Border radius | `20px` (pill) |
| Hover | bg → `0.18` opacity, border → `0.4` opacity |

Default suggestions:

1. "What is Skill-Wanderer?"
2. "The 12 Principles"
3. "Learning Paths"

---

## 8. Message Bubbles

### Layout

- Each message row: `display: flex; gap: 6px`
- User messages: `justify-content: flex-end`
- Assistant messages: `justify-content: flex-start`

### User bubble

| Property | Value |
|---|---|
| Background | `linear-gradient(135deg, #FF6B35, #E85D25)` |
| Text color | `#fff` |
| Border radius | `14px` with `border-bottom-right-radius: 4px` (tail) |
| Shadow | `0 2px 8px rgba(255,107,53,0.2)` |
| Max width | `82%` |
| Font size | `0.8125rem` (13 px) |
| Padding | `0.5rem 0.75rem` |

### Assistant bubble

| Property | Value |
|---|---|
| Background | `rgba(255,255,255,0.06)` |
| Border | `1px solid rgba(255,255,255,0.06)` |
| Text color | `#d0d0d0` |
| Border radius | `14px` with `border-bottom-left-radius: 4px` (tail) |
| Links inside | Color `#FFD93D` (via `:deep(a)`) |
| Max width | `82%` |

### Assistant avatar

- `24×24` circle, `border-radius: 50%`
- Background: `rgba(255,107,53,0.15)`
- Icon: compass SVG `14×14`, color `#FF8C42`
- Left side of assistant messages, aligned to top (`margin-top: 2px`)

---

## 9. Sources Section

Appears below the last assistant message when `sources` are returned.

| Property | Value |
|---|---|
| Margin left | `2rem` (aligns with bubble text, past the avatar) |
| Label | "Discovered Sources" — `10px`, uppercase, wide tracking, 40% opacity |
| Links | `0.6875rem`, color `#FFD93D`, 70% opacity → 100% on hover |
| Icon | Star SVG `10×10`, inline, 60% opacity |
| Overflow | `text-overflow: ellipsis; white-space: nowrap` |

---

## 9½. Clear Chat Button

Appears at the bottom of the messages area when `history` has at least one message and `loading` is false. Clicking it calls `resetChat()` to wipe the conversation and stored session, returning the user to the welcome screen.

| Property | Value |
|---|---|
| Container | Centered (`justify-content: center`), `padding-top: 0.5rem` |
| Background | `rgba(255,107,53,0.08)` |
| Border | `1px solid rgba(255,107,53,0.2)` |
| Text color | `#FF8C42` |
| Font size | `0.6875rem` (11 px) |
| Padding | `5px 12px` |
| Border radius | `20px` (pill) |
| Icon | Refresh SVG `12×12`, inline |
| Hover | bg → `0.18` opacity, border → `0.4` opacity |

---

## 10. Input Bar

| Property | Value |
|---|---|
| Padding | `0.625rem 0.75rem` |
| Background | `rgba(13, 17, 23, 0.95)` |
| Top border | `1px solid rgba(255, 217, 61, 0.08)` |
| Layout | Flex row, gap `0.5rem` |

### Text input

| Property | Value |
|---|---|
| Background | `rgba(255,255,255,0.04)` |
| Border | `1px solid rgba(255,255,255,0.08)` |
| Border radius | `10px` |
| Padding | `0.5rem 0.75rem` |
| Font size | `0.8125rem` |
| Text color | `#e0e0e0` |
| Placeholder | `"Where shall we explore…"` — color `#555`, italic |
| Focus ring | border `rgba(255,107,53,0.4)` + `box-shadow: 0 0 0 2px rgba(255,107,53,0.08)` |
| Max length | `2000` characters |

### Send button (rocket)

| Property | Value |
|---|---|
| Background | `linear-gradient(135deg, #FF6B35, #E85D25)` |
| Size | `38px × 38px` |
| Border radius | `10px` |
| Icon | Rocket SVG `18×18`, white |
| Disabled | 30% opacity, `cursor: not-allowed` |
| Hover (enabled) | `translateY(-1px)` + `box-shadow: 0 4px 12px rgba(255,107,53,0.35)` |

---

## 11. Typing Indicator

Shown inside an assistant bubble while `loading` is true.

- Three dots: `6px` circles, color `#FF8C42`
- Animation: `pathfinder-dot` — scale 0.7 → 1, opacity 0.2 → 1, staggered delay `0s / 0.2s / 0.4s`, period `1.4s`

```css
.pathfinder-typing span {
  width: 6px;
  height: 6px;
  background: #FF8C42;
  border-radius: 50%;
  animation: pathfinder-dot 1.4s infinite both;
}
```

---

## 12. Error State

| Property | Value |
|---|---|
| Position | Bar between messages and input |
| Background | `rgba(255, 107, 107, 0.06)` |
| Border top | `1px solid rgba(255, 107, 107, 0.15)` |
| Text color | `#ff6b6b` |
| Font size | `0.75rem` |
| Padding | `0.5rem 1rem` |

---

## 13. Animations & Motion

| Animation | Duration | Easing | Description |
|---|---|---|---|
| `pathfinder-pulse` | 3 s | `ease-in-out`, infinite | FAB glowing box-shadow pulse |
| `pathfinder-spin` | 12 s | `linear`, infinite | FAB orbit ring rotation |
| `pathfinder-twinkle` | 3–5 s | `ease-in-out`, infinite | Star fade in/out |
| `pathfinder-orbit-bob` | 5–6 s | `ease-in-out`, infinite | Planet icon gentle `translateY(-2px)` bob |
| `pathfinder-dot` | 1.4 s | default, infinite | Typing indicator dot pulse |
| `pathfinder-slide` (enter/leave) | 0.3 s | `ease` | Panel open/close: fade + `translateY(24px) scale(0.92)` |
| Button/card hovers | 0.15–0.25 s | `ease` | Subtle transforms and color shifts |

**General rules:** Keep motion small and subtle. No large or distracting entrance animations.

---

## 14. Responsive / Mobile

| Breakpoint | Behavior |
|---|---|
| `> 480px` | Default floating panel (390×540), FAB at `bottom: 1.5rem; right: 1.5rem` |
| `≤ 480px` | Panel goes **full-screen** (`width: 100vw; height: 100vh; border-radius: 0`), FAB moves to `bottom: 1rem; right: 1rem` |

---

## 15. Typography

All text inside Pathfinder uses the system font stack inherited from the page body:

```
-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif
```

| Element | Size | Weight | Extra |
|---|---|---|---|
| Header title ("Pathfinder") | `text-sm` (~14 px) | **bold** | Gradient text |
| Header subtitle | `10px` | normal | 50% opacity, `tracking-wide` |
| Welcome heading | `text-sm` | **semibold** | — |
| Welcome description | `text-xs` (~12 px) | normal | 60% opacity, `leading-relaxed` |
| Suggestion pills | `0.6875rem` (11 px) | normal | — |
| Chat bubbles | `0.8125rem` (13 px) | normal | `line-height: 1.55` |
| Source label | `10px` | normal | Uppercase, wide tracking, 40% opacity |
| Source links | `0.6875rem` (11 px) | normal | — |
| Input | `0.8125rem` (13 px) | normal | — |
| Error text | `0.75rem` (12 px) | normal | — |

---

## 16. Composable API — `usePathfinder`

Located at `composables/usePathfinder.ts`. Import it with the Nuxt auto-import:

```ts
const { history, loading, error, sendMessage, resetChat } = usePathfinder()
```

### Returned values

| Name | Type | Description |
|---|---|---|
| `history` | `Ref<ChatMessage[]>` | Reactive array of all conversation messages |
| `loading` | `Ref<boolean>` | `true` while waiting for an API response |
| `error` | `Ref<string \| null>` | Error message from the last failed request, or `null` |
| `sendMessage(question)` | `(string) => Promise<ChatResponse \| null>` | Sends a question to the API, appends user+assistant messages to `history`, persists session |
| `resetChat()` | `() => void` | Clears `history` and removes the stored session |

### Types

```ts
interface ChatMessage {
  role: 'user' | 'assistant'
  content: string
}

type Personality = 'librarian' | 'storyteller' | 'admiral'

interface Source {
  title: string
  url: string
  chunk_index: number
  total_chunks: number
  score: number
}

interface ChatResponse {
  answer: string
  sources: Source[]
}
```

### Session persistence

- Mode is controlled by `runtimeConfig.public.pathfinder.sessionStorage` (`"tab"` → `sessionStorage`, anything else → `localStorage`).
- Sessions auto-expire after `sessionExpiryMinutes` (default 30).
- **The expiry countdown resets on every new message.** Each call to `sendMessage()` recalculates `expiresAt` from `Date.now()`, so an active conversation never times out mid-use.
- Storage key: `pathfinder_chat_session`.
- The selected `personality` is persisted as part of the session so the chatbot tone stays consistent across page reloads.
- Users can clear the conversation at any time via the **"Clear chat"** pill button at the bottom of the messages area or the header reset ("New expedition") button. Both call `resetChat()`, which wipes `history` and removes the stored session, returning the panel to the welcome screen so the user can start fresh.

---

## 17. Runtime Config

Set these in `nuxt.config.ts` → `runtimeConfig.public.pathfinder` or via environment variables:

| Config key | Env variable | Default | Description |
|---|---|---|---|
| `apiUrl` | `NUXT_PUBLIC_PATHFINDER_API_URL` | `http://localhost:8000` | Backend API base URL |
| `domains` | `NUXT_PUBLIC_PATHFINDER_DOMAINS` | `skill-wanderer.com,wanderings.skill-wanderer.com,dojo.skill-wanderer.com` | Comma-separated domain filter list |
| `sessionExpiryMinutes` | `NUXT_PUBLIC_PATHFINDER_SESSION_EXPIRY_MINUTES` | `30` | Minutes before stored session expires |
| `sessionStorage` | `NUXT_PUBLIC_PATHFINDER_SESSION_STORAGE` | `browser` | `"tab"` for sessionStorage, anything else for localStorage |

---

## 18. Chat API Reference

### Base URL

```
http://localhost:8000
```

All chat endpoints are under `/api`.

### Send a Message — `POST /api/chat`

**Request**

```http
POST /api/chat
Content-Type: application/json
```

| Field      | Type              | Required | Description                                           |
| ---------- | ----------------- | -------- | ----------------------------------------------------- |
| `question` | `string`          | Yes      | The user's question (1–2 000 chars).                  |
| `domain`   | `string \| null`  | No       | Filter results to a single domain (e.g. `"example.com"`). Use `domains` for multiple. |
| `domains`  | `string[] \| null`| No       | Filter results to one or more domains.                |
| `url`      | `string \| null`  | No       | Filter results to a specific URL prefix.              |
| `personality` | `string \| null` | No       | Chatbot personality: `"librarian"` (Lyra the Archivist), `"storyteller"` (Nova the Weaver), or `"admiral"` (Admiral Orion). Omit for the default Pathfinder voice. |
| `history`  | `array`           | No       | Previous conversation turns, oldest first (max 50).   |

> **Note:** `domain` and `domains` can be used together — they are merged into a single list.

> **Personalities:** The optional `personality` field changes the tone and style of Pathfinder's responses:
>
> | Value | Character | Best for | Vibe | Catchphrase |
> |---|---|---|---|---|
> | `"librarian"` | 📚 **Lyra the Archivist** | LMS / learning content | Keeper of ancient star charts and deep-space data — calm, scholarly, precise, and timeless | *"The records are clear. Here is the knowledge you seek."* |
> | `"storyteller"` | ✍️ **Nova the Weaver** | Blog posts | Traveler who has seen a thousand suns — warm, vivid, narrative, turns every fact into a fable | *"Every star has a story, and this one begins with your question..."* |
> | `"admiral"` | 🌌 **Admiral Orion** | Whole site | High-ranking officer overseeing the fleet — confident, commanding, mission-focused | *"Course plotted. Scanning the sector for answers."* |
>
> When omitted (or `null`), Pathfinder uses its default neutral assistant voice.

Each item in `history`:

| Field     | Type                         | Description                  |
| --------- | ---------------------------- | ---------------------------- |
| `role`    | `"user"` \| `"assistant"`    | Who sent the message.        |
| `content` | `string`                     | The message text (1–4 000 chars). |

**Example request**

```json
{
  "question": "How do I enable two-factor auth?",
  "domains": ["help.example.com", "docs.example.com"],
  "personality": "librarian",
  "history": [
    { "role": "user", "content": "How do I reset my password?" },
    { "role": "assistant", "content": "Go to Settings > Security and click 'Reset password'." }
  ]
}
```

**Response `200 OK`**

```json
{
  "answer": "To reset your password, go to Settings > Security …",
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

| Field                   | Type     | Description                                      |
| ----------------------- | -------- | ------------------------------------------------ |
| `answer`                | `string` | The generated answer.                             |
| `sources`               | `array`  | Source chunks used to build the answer.            |
| `sources[].title`       | `string` | Page title.                                       |
| `sources[].url`         | `string` | Original URL.                                     |
| `sources[].chunk_index` | `number` | Chunk index within the page.                      |
| `sources[].total_chunks`| `number` | Total chunks for that page.                       |
| `sources[].score`       | `number` | Similarity score (higher = more relevant).        |

---

## 19. CSS Class Reference

Quick-look table of every Pathfinder-specific CSS class. All classes are **scoped** to `PathfinderChat.vue`.

| Class | Element | Notes |
|---|---|---|
| `.pathfinder-fab` | FAB button | Compass, fixed bottom-right |
| `.pathfinder-fab-ring` | Orbit ring on FAB | Dashed spinning circle |
| `.pathfinder-panel` | Chat panel container | Dark card, flexbox column |
| `.pathfinder-stars` | Starfield wrapper | Absolute, pointer-events none |
| `.pathfinder-star` | Individual star | 2 px dot with twinkle animation |
| `.pathfinder-header` | Header bar | Gradient bg, flex row |
| `.pathfinder-planet-icon` | Planet SVG in header | Bobbing animation |
| `.pathfinder-title-text` | "Pathfinder" text | Gradient clip text |
| `.pathfinder-header-btn` | Header icon buttons | Reset & close |
| `.pathfinder-messages` | Scrollable message area | flex: 1, overflow-y auto |
| `.pathfinder-welcome` | Welcome empty state | Centered, faded text |
| `.pathfinder-welcome-planet` | Large planet icon | Bob animation |
| `.pathfinder-suggestions` | Suggestion pill container | Flex-wrap row |
| `.pathfinder-suggestion` | Individual suggestion pill | Pill shape, orange border |
| `.pathfinder-msg` | Message row | Flex, gap 6 px |
| `.pathfinder-msg-user` | User row modifier | Align right |
| `.pathfinder-msg-assistant` | Assistant row modifier | Align left |
| `.pathfinder-avatar` | Assistant avatar circle | 24 px, compass icon |
| `.pathfinder-bubble` | Message bubble | Rounded, max-width 82 % |
| `.pathfinder-typing` | Typing indicator bubble | Three animated dots |
| `.pathfinder-sources` | Sources container | Below assistant message |
| `.pathfinder-source-link` | Individual source link | Star icon + ellipsis |
| `.pathfinder-clear-chat` | Clear chat wrapper | Centered, below messages |
| `.pathfinder-clear-btn` | Clear chat pill button | Pill shape, orange border, icon + text |
| `.pathfinder-error` | Error bar | Red tinted strip |
| `.pathfinder-input-bar` | Input form wrapper | Bottom of panel |
| `.pathfinder-input` | Text input field | Rounded, dark bg |
| `.pathfinder-send-btn` | Rocket send button | Orange gradient square |
| `.pathfinder-slide-*` | Vue transition classes | Enter/leave for panel |

**Error responses**

| Status | Meaning                                    |
| ------ | ------------------------------------------ |
| `422`  | Validation error (e.g. empty question, invalid `personality` value). |
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
async function askPathfinder(question, { domain, domains, personality, history = [] } = {}) {
  const res = await fetch("http://localhost:8000/api/chat", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question, domain, domains, personality, history }),
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

// Usage — with a personality
reply = await askPathfinder("Explain the 12 Principles", {
  personality: "storyteller",
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

type Personality = "librarian" | "storyteller" | "admiral";

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
      opts?: { domain?: string; domains?: string[]; personality?: Personality }
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
            personality: opts?.personality ?? null,
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

### Nuxt 3 (Skill-Wanderer setup)

The Skill-Wanderer chatbot uses a multi-domain configuration so Pathfinder searches across all three properties at once.

**Environment variable (`.env`)**

```bash
# Comma-separated list of domains Pathfinder searches across
NUXT_PUBLIC_PATHFINDER_DOMAINS=skill-wanderer.com,wanderings.skill-wanderer.com,dojo.skill-wanderer.com
```

**Runtime config (`nuxt.config.ts`)**

```ts
runtimeConfig: {
  public: {
    pathfinder: {
      apiUrl: process.env.NUXT_PUBLIC_PATHFINDER_API_URL || 'http://localhost:8000',
      domains: process.env.NUXT_PUBLIC_PATHFINDER_DOMAINS || 'skill-wanderer.com,wanderings.skill-wanderer.com,dojo.skill-wanderer.com',
      sessionExpiryMinutes: Number(process.env.NUXT_PUBLIC_PATHFINDER_SESSION_EXPIRY_MINUTES) || 30,
      sessionStorage: process.env.NUXT_PUBLIC_PATHFINDER_SESSION_STORAGE || 'browser',
    },
  },
},
```

**Composable (`composables/usePathfinder.ts`)**

```ts
export function usePathfinder() {
  const config = useRuntimeConfig()
  const { apiUrl, domains: domainsRaw, sessionExpiryMinutes, sessionStorage: storageMode } =
    config.public.pathfinder as {
      apiUrl: string
      domains: string
      sessionExpiryMinutes: number
      sessionStorage: string
    }

  // Parse the comma-separated string into an array
  const domains = domainsRaw.split(',').map(d => d.trim()).filter(Boolean)

  async function sendMessage(question: string) {
    const res = await $fetch(`${apiUrl}/api/chat`, {
      method: 'POST',
      body: {
        question,
        domains,                          // sends ["skill-wanderer.com", "wanderings.skill-wanderer.com", "dojo.skill-wanderer.com"]
        personality: personality.value,    // current personality (or null for default)
        history: history.value.slice(-50),
      },
    })
    // …
  }
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
