# J.A.R.V.I.S. 🧠

J.A.R.V.I.S. is an advanced, highly sophisticated AI assistant featuring a Python FastAPI backend and a dynamic React + WebGL frontend. It features a conversational AI core with tool-calling capabilities, audio-reactive 3D visualizations, and seamless voice interactions.

---

## ✨ Key Features

- **Interactive 3D Interface**: A mesmerizing, audio-reactive WebGL plasma sphere built with Three.js.
- **Voice Interactions**: Features browser-native Text-to-Speech (TTS) for natural spoken responses, alongside Wake Word detection ("JARVIS").
- **Agentic Tool Calling**: A robust modular tool framework allowing the AI to execute functions dynamically.
- **Real-time Web Search**: Integrates **Tavily** (and DuckDuckGo) for high-accuracy, AI-optimized web searches. J.A.R.V.I.S. intelligently determines when to search the web for freshness-sensitive queries (e.g., news, current events, latest updates).
- **Flexible AI Providers**: Built with a provider abstraction layer. Currently configured with **Groq** for high-speed inference.

## 🏗 Architecture

```
JARVIS/
├── frontend/               # React 19 + Vite + Three.js Client
│   ├── src/
│   │   ├── component/      # UI, 3D Blob, Chat Interface
│   │   └── services/       # API Integration
├── backend/                # FastAPI Application
│   ├── app/
│   │   ├── api/            # REST Endpoints
│   │   ├── config/         # System Prompt & Env Config
│   │   ├── providers/      # AI Provider Integrations (Groq, etc.)
│   │   └── services/       # Core Logic & Tool Registry (Search, etc.)
│   └── tests/              # Comprehensive Pytest Suite
└── package.json            # Root task runner
```

## 🚀 Getting Started

### Prerequisites

- **Python 3.13+**
- **Node.js 18+** & npm

### 1. Installation

You can install dependencies for both the frontend and backend from the root directory:

```bash
npm run install-all
```

Alternatively, you can install them manually:
* **Backend:** `cd backend && python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt`
* **Frontend:** `cd frontend && npm install`

### 2. Configuration

Navigate to the `backend` directory and configure your environment variables:

```bash
cd backend
cp .env.example .env
```

Edit the `.env` file and add your API keys:
```env
GROQ_API_KEY="your_groq_api_key_here"
TAVILY_API_KEY="your_tavily_api_key_here"
```

### 3. Running the Application

Start both the frontend and backend servers concurrently from the root directory:

```bash
npm start
```

- **Frontend** will be available at: `http://localhost:5173`
- **Backend API** will be available at: `http://localhost:8000`

## 🧪 Testing

The backend includes a comprehensive test suite covering the AI manager, tool registry, and specific tool implementations.

```bash
cd backend
source .venv/bin/activate
pytest tests/ -v
```

## 🛡️ License

This project is licensed under standard open-source terms. Created and developed by Sandilya Kavi.
