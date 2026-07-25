# Jarvis AI Backend - Foundation (Sprint 1)

This is the backend foundation for Project JARVIS, built using **FastAPI** with a Service-Oriented Architecture (SOA) modular folder structure.

## Directory Structure

```
backend/
├── app/
│   ├── api/          # API routers and endpoints
│   ├── config/       # Configuration management (Pydantic settings)
│   ├── services/     # Core business logic and service layer
│   ├── providers/    # Third-party integrations (e.g., LLM providers, tools)
│   ├── database/     # Database session setup and connection setup
│   ├── models/       # Database ORM models
│   ├── schemas/      # Pydantic data schemas
│   ├── utils/        # Shared helper/utility functions
│   └── main.py       # FastAPI application initialization & middleware
├── tests/            # Test suite
├── requirements.txt  # Python package requirements
├── .env.example      # Sample environment configuration file
├── .gitignore        # Git ignore rules for Python
└── README.md         # Documentations & run guide
```

---

## Getting Started

### 1. Prerequisites
- Python 3.10+
- `pip` (Python package manager)

### 2. Setup Virtual Environment
Run the following commands inside the `backend` directory:

```bash
# Create a virtual environment
python3 -m venv .venv

# Activate the virtual environment
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Environment Configuration
Copy `.env.example` to `.env` and adjust the variables:
```bash
cp .env.example .env
```

### 5. Running the Application
To run the server in development mode with hot-reloading:
```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```
Open [http://localhost:8000](http://localhost:8000) in your browser. Swagger documentation is available at [http://localhost:8000/docs](http://localhost:8000/docs).

### 6. Running Tests
Run tests using `pytest`:
```bash
pytest
```
