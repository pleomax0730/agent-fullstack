# AI Elements + Python Backend Integration

This project demonstrates how to use Vercel's **AI Elements** components with a custom **Python FastAPI backend** instead of the Vercel AI SDK backend.

## Quick Start

### 1. Start the Backend (Python)

```bash
# In the root folder (ai-elements-playground)
uv run python fastapi_sse_backend.py
```

Backend will run at: `http://localhost:8000`

### 2. Start the Frontend (Next.js)

```bash
# In the frontend folder (ai-chatbot)
cd ai-chatbot
npm run dev
```

Frontend will run at: `http://localhost:3000`

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│  Frontend (Next.js + AI Elements)                          │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  useChat hook from @ai-sdk/react                    │   │
│  │  - Conversation, Message, Reasoning, Sources, etc.  │   │
│  └─────────────────────────────────────────────────────┘   │
│                           │                                │
│                           │ SSE (Server-Sent Events)       │
│                           │ UI Message Stream Protocol     │
│                           ▼                                │
├─────────────────────────────────────────────────────────────┤
│  Backend (FastAPI + LlamaIndex)                            │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  /api/chat endpoint                                 │   │
│  │  - Implements UI Message Stream Protocol            │   │
│  │  - Streams text, reasoning, tool calls              │   │
│  └─────────────────────────────────────────────────────┘   │
│                           │                                │
│                           ▼                                │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  LlamaIndex + Google GenAI (Gemini)                 │   │
│  │  - Streaming chat with tools                        │   │
│  │  - Thinking/reasoning support                       │   │
│  └─────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

## UI Message Stream Protocol

The key to making AI Elements work with a custom backend is implementing the **UI Message Stream Protocol**. This is the SSE format that `useChat` expects.

### Required Header

```
x-vercel-ai-ui-message-stream: v1
```

### Message Types

| Type | Description |
|------|-------------|
| `start` | Message start with `messageId` |
| `text-start` | Start of text content |
| `text-delta` | Incremental text chunk |
| `text-end` | End of text content |
| `reasoning-start` | Start of reasoning/thinking |
| `reasoning-delta` | Incremental reasoning chunk |
| `reasoning-end` | End of reasoning |
| `tool-input-start` | Start of tool call |
| `tool-input-delta` | Tool call arguments (input) |
| `tool-input-available` | End of tool call input |
| `tool-output-available` | Tool execution result |
| `source-url` | Citation/source URL |
| `finish-step` | End of a step (LLM call) |
| `finish` | End of message |
| `[DONE]` | Stream termination |

### Example Stream

```
data: {"type":"start","messageId":"abc123"}

data: {"type":"reasoning-start","id":"r1"}

data: {"type":"reasoning-delta","id":"r1","delta":"Let me think..."}

data: {"type":"reasoning-end","id":"r1"}

data: {"type":"text-start","id":"t1"}

data: {"type":"text-delta","id":"t1","delta":"Hello! "}

data: {"type":"text-delta","id":"t1","delta":"How can I help?"}

data: {"type":"text-end","id":"t1"}

data: {"type":"finish-step"}

data: {"type":"finish"}

data: [DONE]
```

## Stream Lifecycle & Control Flow

### 1. Standard Chat Flow (No Tools)

```
[Start]
  │
  ├── data: {"type":"start", "messageId":"..."}
  │
  ├── [Step 1: LLM Response]
  │     ├── data: {"type":"reasoning-start", "id":"r1"}
  │     ├── data: {"type":"reasoning-delta", "id":"r1", "delta":"思考中..."}
  │     ├── data: {"type":"reasoning-end", "id":"r1"}
  │     │
  │     ├── data: {"type":"text-start", "id":"t1"}
  │     ├── data: {"type":"text-delta", "id":"t1", "delta":"你好！"}
  │     └── data: {"type":"text-end", "id":"t1"}
  │
  ├── data: {"type":"finish-step"}  ← End of Step 1
  │
  ├── data: {"type":"finish"}       ← End of Message
  │
  └── data: [DONE]                  ← Close Connection
```

### 2. Multi-Step Flow (With Tool Calls)

When the model decides to call a tool, the flow involves multiple steps (loops).

```
[Start]
  │
  ├── data: {"type":"start", "messageId":"msg-123"}
  │
  ├── [Step 1: Tool Invocation]
  │     ├── data: {"type":"tool-input-start", "toolCallId":"call-1", "toolName":"check_flight"}
  │     ├── data: {"type":"tool-input-delta", "toolCallId":"call-1", "inputTextDelta":"{\"flight\":\"AA100\"}"}
  │     ├── data: {"type":"tool-input-available", "toolCallId":"call-1", "toolName":"check_flight", "input":{...}}
  │     │
  │     ├── (Backend executes tool...)
  │     │
  │     └── data: {"type":"tool-output-available", "toolCallId":"call-1", "output":{...}}
  │
  ├── data: {"type":"finish-step"}  ← End of Step 1 (Loop continues)
  │
  ├── [Step 2: LLM Response based on Tool Result]
  │     ├── data: {"type":"text-start", "id":"t1"}
  │     ├── data: {"type":"text-delta", "id":"t1", "delta":"航班 AA100 目前狀態..."}
  │     └── data: {"type":"text-end", "id":"t1"}
  │
  ├── data: {"type":"finish-step"}  ← End of Step 2
  │
  ├── data: {"type":"finish"}       ← End of Message
  │
  └── data: [DONE]                  ← Close Connection
```

### 3. Event Hierarchy

| Event | Scope | Purpose |
|-------|-------|---------|
| `[DONE]` | **Connection** | Tells browser/client to close the HTTP connection. |
| `finish` | **Message** | Marks the entire assistant message as complete. |
| `finish-step` | **Step** | Marks one LLM interaction cycle complete (e.g., after tool call or final text). |
| `*-end` | **Content Block** | Marks a specific content block (text/reasoning) as complete. |

### 4. `id` 欄位說明

每個內容區塊 (text, reasoning) 都有一個 `id`，用來關聯同一區塊的 `start`/`delta`/`end` 事件。

```
後端產生 ID                           前端處理
─────────────                        ─────────
text_part_id = uuid4()               
  │                                  
  ├─► {"type":"text-start","id":"abc"}    → 建立 part {id:"abc", text:""}
  ├─► {"type":"text-delta","id":"abc","delta":"Hi"}  → 找到 id=abc，附加文字
  ├─► {"type":"text-delta","id":"abc","delta":"!"}   → part.text = "Hi!"
  └─► {"type":"text-end","id":"abc"}      → 標記完成
```

### 5. 後端程式碼對應 (`fastapi_sse_backend.py`)

| 行號 | 程式碼 | 說明 |
|------|--------|------|
| 90-92 | `create_message_start()` | 產生 `start` 事件 |
| 95-107 | `create_text_start/delta/end()` | 產生文字串流事件 |
| 110-122 | `create_reasoning_start/delta/end()` | 產生思考串流事件 |
| 125-140 | `create_tool_input_*()` | 產生工具輸入事件 |
| 143-145 | `create_tool_output()` | 產生工具輸出事件 |
| 150-157 | `create_finish_step/finish()` | 產生結束事件 |
| 235 | `yield sse_event(create_message_start(...))` | 串流開始 |
| 274-277 | `yield sse_event(create_reasoning_*)` | 串流思考內容 |
| 283-289 | `yield sse_event(create_text_*)` | 串流文字內容 |
| 322-337 | `yield sse_event(create_tool_*)` | 串流工具調用 |
| 349 | `yield sse_event(create_finish_step())` | 工具步驟結束，繼續迴圈 |
| 356-358 | `yield finish_step + finish + [DONE]` | 最終結束 |

### 6. 完整流程圖

```
┌─────────────────────────────────────────────────────────────────────┐
│                        用戶發送訊息                                   │
│                    "幫我查 AA100 班機狀態"                            │
└────────────────────────────┬────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│  POST /api/chat                                                     │
│  body: { messages: [...] }                                          │
└────────────────────────────┬────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│  stream_chat_response()                                             │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │ yield: start                                                  │  │
│  └───────────────────────────────────────────────────────────────┘  │
│                             │                                       │
│                             ▼                                       │
│  ┌─ for step in range(max_steps): ─────────────────────────────┐   │
│  │                                                              │   │
│  │  ┌─ Step 1: LLM 決定調用工具 ────────────────────────────┐  │   │
│  │  │  (LLM: "我需要查詢航班資訊")                           │  │   │
│  │  │  yield: tool-input-start                              │  │   │
│  │  │  yield: tool-input-delta                              │  │   │
│  │  │  yield: tool-input-available                          │  │   │
│  │  │                                                        │  │   │
│  │  │  → 執行 check_flight("AA100")                         │  │   │
│  │  │  → 結果: {"status": "delayed", ...}                   │  │   │
│  │  │                                                        │  │   │
│  │  │  yield: tool-output-available                         │  │   │
│  │  │  yield: finish-step                                   │  │   │
│  │  │  continue ─────────────────────────────────────────────┼──┤   │
│  │  └────────────────────────────────────────────────────────┘  │   │
│  │                                                              │   │
│  │  ┌─ Step 2: LLM 根據工具結果回應 ────────────────────────┐  │   │
│  │  │  (LLM: "根據查詢結果，AA100 航班...")                  │  │   │
│  │  │  yield: reasoning-start                               │  │   │
│  │  │  yield: reasoning-delta (思考內容...)                 │  │   │
│  │  │  yield: reasoning-end                                 │  │   │
│  │  │  yield: text-start                                    │  │   │
│  │  │  yield: text-delta ("航班 AA100...")                  │  │   │
│  │  │  yield: text-end                                      │  │   │
│  │  │  yield: finish-step                                   │  │   │
│  │  │  break ────────────────────────────────────────────────┼──┤   │
│  │  └────────────────────────────────────────────────────────┘  │   │
│  │                                                              │   │
│  └──────────────────────────────────────────────────────────────┘   │
│                             │                                       │
│                             ▼                                       │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │ yield: finish                                                 │  │
│  │ yield: [DONE]                                                 │  │
│  └───────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│  前端 useChat 接收 SSE 事件                                          │
│  → 解析每個事件，更新 messages.parts                                 │
│  → UI 即時顯示思考過程、工具調用、最終回答                           │
└─────────────────────────────────────────────────────────────────────┘
```

## Setup from Scratch

### Backend (Python)

1. Install dependencies:

```bash
uv sync
# or
pip install -r requirements.txt
```

2. Configure credentials:
   - Create `service_account.json` with your Google Cloud/VertexAI credentials
   - Or set up environment variables

### Frontend (Next.js)

1. Create a new Next.js project:

```bash
npx create-next-app@latest ai-chatbot && cd ai-chatbot
```

2. Install AI Elements & AI SDK:

```bash
npx ai-elements@latest add conversation message prompt-input reasoning loader tool
npm install @ai-sdk/react ai lucide-react
```

3. Set the backend URL in `.env.local`:

```
NEXT_PUBLIC_BACKEND_URL=http://localhost:8000
```

4. Use the `frontend_example.tsx` as a reference to configure `useChat`.

## Project Structure

### Root Folder (Python Backend)

```
ai-elements-playground/
├── fastapi_sse_backend.py    # 🐍 主要後端程式 (FastAPI + SSE)
├── llm_stream_tool_demo.py   # 🧪 原始 LlamaIndex 測試腳本
├── service_account.json      # 🔑 Google Cloud 憑證檔案
├── pyproject.toml            # 📦 Python 套件定義
├── uv.lock                   # 🔒 Python 套件鎖定版本
└── ai-chatbot/               # 📁 前端專案資料夾 (見下方)
```

### Frontend Folder (`ai-chatbot/`)

```
ai-chatbot/
│
├── 📁 app/                        # Next.js App Router (頁面路由)
│   ├── page.tsx                   # 🏠 首頁 - 你的聊天機器人主程式
│   ├── layout.tsx                 # 📐 全站佈局 (共用的 HTML 結構)
│   ├── globals.css                # 🎨 全域 CSS 樣式
│   └── favicon.ico                # 🖼️ 網站圖示
│
├── 📁 components/                 # 可重用的 UI 元件
│   │
│   ├── 📁 ai-elements/            # ⭐ AI Elements 元件 (核心!)
│   │   ├── conversation.tsx       # 💬 對話容器 - 包住所有訊息
│   │   ├── message.tsx            # 📝 單則訊息 - 顯示 user/assistant 訊息
│   │   ├── reasoning.tsx          # 🧠 思考過程 - 顯示 AI 的推理
│   │   ├── prompt-input.tsx       # ⌨️ 輸入框 - 用戶輸入訊息的地方
│   │   ├── loader.tsx             # ⏳ 載入動畫 - 等待回應時顯示
│   │   ├── tool.tsx               # 🔧 工具顯示 - 顯示工具調用結果
│   │   ├── sources.tsx            # 📚 來源引用 - 顯示參考資料
│   │   ├── code-block.tsx         # 💻 程式碼區塊 - 格式化程式碼
│   │   └── ...                    # 其他 AI 相關元件
│   │
│   └── 📁 ui/                     # 🎨 基礎 UI 元件 (shadcn/ui)
│       ├── button.tsx             # 按鈕
│       ├── input.tsx              # 輸入框
│       ├── card.tsx               # 卡片
│       ├── badge.tsx              # 標籤徽章
│       ├── collapsible.tsx        # 可折疊區塊
│       └── ...                    # 其他基礎元件
│
├── 📁 lib/                        # 工具函式庫
│   └── utils.ts                   # 🛠️ 通用工具函式 (如 className 合併)
│
├── 📁 public/                     # 靜態資源 (圖片、字體等)
│   └── *.svg                      # SVG 圖示
│
├── .env.local                     # 🔐 環境變數 (後端 URL 設定在這)
├── package.json                   # 📦 npm 套件定義 & 啟動腳本
├── tsconfig.json                  # ⚙️ TypeScript 設定
├── next.config.ts                 # ⚙️ Next.js 設定
├── components.json                # ⚙️ shadcn/ui 設定
└── node_modules/                  # 📚 已安裝的套件 (自動產生)
```

### 元件關係圖

```
page.tsx (主頁面)
│
├── useChat()                      # AI SDK 的 React Hook
│   └── 管理 messages[], sendMessage(), status 等
│
├── <Conversation>                 # 對話容器
│   └── <ConversationContent>
│       ├── <Message>              # 每則訊息
│       │   ├── <MessageContent>
│       │   │   └── <MessageResponse> (文字內容)
│       │   └── <MessageActions>   # 複製、重試按鈕
│       │
│       ├── <Reasoning>            # AI 思考過程
│       │   ├── <ReasoningTrigger> # 展開/收合按鈕
│       │   └── <ReasoningContent> # 思考內容
│       │
│       ├── <Tool>                 # 工具調用
│       │   ├── <ToolHeader>       # 工具名稱 + 狀態
│       │   └── <ToolContent>      # 輸入參數 + 輸出結果
│       │
│       └── <Loader>               # 載入中動畫
│
└── <PromptInput>                  # 輸入區域
    ├── <PromptInputTextarea>      # 文字輸入框
    └── <PromptInputSubmit>        # 送出按鈕
```

### 關鍵檔案說明

| 檔案 | 功能 |
|------|------|
| `app/page.tsx` | **主程式** - 所有聊天邏輯都在這裡，包括 useChat 設定和 UI 組裝 |
| `components/ai-elements/*.tsx` | **AI 元件** - 用來顯示對話、思考、工具等 AI 相關內容 |
| `components/ui/*.tsx` | **基礎元件** - 按鈕、輸入框等基本 UI 元素 (來自 shadcn/ui) |
| `.env.local` | **環境設定** - 設定 `NEXT_PUBLIC_BACKEND_URL` 指向 Python 後端 |
| `package.json` | **套件清單** - 定義專案用到的所有 npm 套件和啟動指令 |

## Features

- ✅ Streaming text responses
- ✅ Reasoning/thinking display (for models like Gemini with thinking)
- ✅ Tool calls with execution results
- ✅ Compatible with all AI Elements components
- ✅ CORS configured for local development

## Customization

### Adding More Tools

Add tools in `fastapi_sse_backend.py`:

```python
def my_custom_tool(param: str) -> dict:
    """Description of what the tool does."""
    return {"result": "..."}

TOOLS = [
    FunctionTool.from_defaults(fn=my_custom_tool),
    # ... other tools
]
```

### Changing the LLM

Modify the `get_llm()` function to use a different model or provider.

### Adding Sources/Citations

Use the `source-url` event type:

```python
yield sse_event({"type": "source-url", "url": "https://example.com"})
```

## References

- [AI Elements Documentation](https://ai-elements.dev)
- [Vercel AI SDK](https://sdk.vercel.ai/docs)
- [LlamaIndex](https://docs.llamaindex.ai)
