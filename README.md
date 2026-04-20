# 🚀 Blog Writer Multi-Agent AI

A custom multi-agent system designed to autonomously create high-quality, research-driven blogs. Using **LangChain**, **Qwen3-1.7B** (default, with Google Gemini ready), and **Serper Web Search**, it automates planning, writing, and editing to deliver human-like blogs.

![Architecture](https://img.shields.io/badge/Architecture-Multi--Agent-blueviolet)
![LangChain](https://img.shields.io/badge/LangChain-0.3-green)
![Qwen3](https://img.shields.io/badge/Default_Model-Qwen3--1.7B-orange)
![Next.js](https://img.shields.io/badge/Frontend-Next.js-black)

## ✨ Features

- **🤖 3 AI Agents**: Planner → Writer → Editor pipeline
- **📄 Document Upload**: Upload PDF, DOCX, TXT, MD as reference materials
- **🔍 Web Search**: Automatic web research via Serper API
- **🔄 Switchable LLM**: Default Qwen3-1.7B, ready for Google Gemini
- **🎨 Premium UI**: Dark theme with glassmorphism and animations
- **📋 Multi-view Output**: View Final blog, Content Plan, and Draft separately

## 📁 Project Structure

```
Blog-writer-multi-agent/
├── server/                     # FastAPI Backend
│   ├── app/
│   │   ├── main.py             # API endpoints
│   │   ├── llm_config.py       # LLM provider factory (HF/Google)
│   │   ├── agents.py           # Agent chain definitions
│   │   ├── tasks.py            # Prompt templates
│   │   ├── crew.py             # Sequential orchestration
│   │   └── document_handler.py # File upload & text extraction
│   ├── .env.example
│   └── requirements.txt
│
├── client/bloggpt/             # Next.js Frontend
│   ├── app/                    # App router pages
│   ├── components/
│   │   └── BlogGenerator.js    # Main UI component
│   └── actions/
│       └── generateBlog.js     # API calls
│
├── .gitignore
└── README.md
```

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- Node.js 18+
- HuggingFace account ([get token](https://huggingface.co/settings/tokens))
- Serper API key ([get free key](https://serper.dev/)) — optional

### 1. Backend Setup

```bash
cd server

# Create virtual environment
python -m venv venv
venv\Scripts\activate      # Windows
# source venv/bin/activate # Linux/Mac

# Install dependencies
pip install -r requirements.txt

# Configure environment
copy .env.example .env     # Windows
# cp .env.example .env     # Linux/Mac

# Edit .env with your API keys
# Required: HF_TOKEN
# Optional: SERPER_API_KEY, GOOGLE_API_KEY

# Run server
uvicorn app.main:app --host 127.0.0.1 --port 8002 --reload
```

### 2. Frontend Setup

```bash
cd client/bloggpt

# Install dependencies
npm install

# Run dev server
npm run dev
```

### 3. Open Browser

Navigate to **http://localhost:3000** and start generating blogs!

## 🔧 LLM Configuration

### Default: HuggingFace (Qwen3-1.7B)

```env
LLM_PROVIDER=huggingface
HF_TOKEN=hf_your_token_here
HF_MODEL=Qwen/Qwen3-1.7B
```

### Switch to Google AI Studio (Gemini)

```env
LLM_PROVIDER=google
GOOGLE_API_KEY=your_google_api_key
GOOGLE_MODEL=gemini-2.0-flash
```

## 🧠 How It Works

1. **User Input**: Enter a topic + optionally upload reference documents
2. **Web Search**: Serper API fetches latest info on the topic
3. **Planner Agent**: Creates structured content outline with SEO keywords
4. **Writer Agent**: Drafts the full blog post following the outline
5. **Editor Agent**: Polishes grammar, tone, structure, and formatting
6. **Output**: Final blog in Markdown, viewable with Plan and Draft tabs

## 📄 Document Upload

Upload reference documents to provide context for blog generation:

| Format | Extension | Description |
|--------|-----------|-------------|
| PDF    | `.pdf`    | Research papers, reports |
| Word   | `.docx`   | Documents, notes |
| Text   | `.txt`    | Plain text files |
| Markdown | `.md`   | Markdown documents |

- Max 5 files per request
- Max 10MB per file
- Documents are used as reference context, not copied verbatim

## 🛠 Tech Stack

| Component | Technology |
|-----------|------------|
| Agent Framework | LangChain |
| Default LLM | Qwen3-1.7B (HuggingFace) |
| Ready LLM | Google Gemini (AI Studio) |
| Backend | FastAPI + Python |
| Frontend | Next.js + React |
| Web Search | Serper API |
| Styling | Vanilla CSS (Premium Dark Theme) |

## 📝 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/generate-blog/` | Generate blog (multipart form, supports file upload) |
| `POST` | `/generate-blog-json/` | Generate blog (JSON body, no files) |
| `GET` | `/health` | Health check & LLM info |
| `GET` | `/config` | Current LLM configuration |

## 📜 License

MIT
