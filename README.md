# 💧 Water Quality RAG Chatbot

A production-ready **Retrieval-Augmented Generation (RAG)** chatbot for water quality monitoring. Built with **Groq LLM**, **LangChain**, **FAISS**, and **HuggingFace embeddings** — architected for seamless React dashboard integration via **FastAPI**.

---


## 📁 Folder Structure

```
water_quality_rag/
├── app/
│   ├── core/
│   │   ├── config.py          # Pydantic settings (reads .env)
│   │   └── logger.py          # Loguru logging setup
│   │
│   ├── data/
│   │   ├── water_quality_data.csv   # Your dataset (replace with real data)
│   │   ├── who_guidelines.py        # WHO knowledge base (static)
│   │   └── faiss_index/             # Auto-created FAISS index (gitignored)
│   │
│   ├── services/
│   │   ├── data_loader.py     # CSV → LangChain Documents
│   │   ├── vector_store.py    # FAISS build/load/retrieve
│   │   └── rag_pipeline.py    # Core RAG chain (Groq + LangChain)
│   │
│   ├── utils/
│   │   └── topic_guard.py     # Keyword filter for off-topic queries
│   │
│   ├── api/
│   │   └── main.py            # FastAPI app with all endpoints
│   │
│   └── ui/
│       └── streamlit_app.py   # Streamlit prototype chat UI
│
├── scripts/
│   └── build_index.py         # Standalone FAISS index builder
│
├── tests/
│   └── test_services.py       # Unit tests (pytest)
│
├── .env
├── requirements.txt           # All Python dependencies
├── .gitignore                 # Excludes .env, FAISS index, venv
└── README.md
```

---

## 🚀 Quick Start

### 1. Clone and set up environment

```bash
git clone https://github.com/jiya-19/water_quality_rag.git
cd water_quality_rag

# Create virtual environment
python -m venv venv
source venv/bin/activate        # macOS/Linux
# venv\Scripts\activate         # Windows

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure environment variables

# Edit .env and add your GROQ_API_KEY

Get your free Groq API key at: https://console.groq.com/



### 4. Build the FAISS index

```bash
python scripts/build_index.py
# Force rebuild:  python scripts/build_index.py --force
# Custom CSV:     python scripts/build_index.py --csv /path/to/data.csv
```

### 5. Run the Streamlit prototype

```bash
streamlit run app/ui/streamlit_app.py
```

Open http://localhost:8501 in your browser.

### 6. Run the FastAPI server

```bash
uvicorn app.api.main:app --host 0.0.0.0 --port 8000 --reload
```

Open http://localhost:8000/docs for the interactive API documentation.

---

## 🔌 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/chat` | Ask a water quality question |
| `POST` | `/compare` | Compare two water bodies |
| `GET` | `/waterbody/{name}` | Get data for a specific water body |
| `GET` | `/wqi/{location}` | Get WQI data for a location |
| `GET` | `/waterbodies` | List all water bodies (with filters) |
| `GET` | `/health` | Health check |
| `POST` | `/admin/rebuild-index` | Rebuild the FAISS index |

### Example: Chat

```bash
curl -X POST "http://localhost:8000/chat" \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the WQI of Sabarmati River?"}'
```

### Example: Compare

```bash
curl -X POST "http://localhost:8000/compare" \
  -H "Content-Type: application/json" \
  -d '{"body_a": "Sabarmati River", "body_b": "Tapi River"}'
```

### React Integration Example

```javascript
// In your React dashboard component:
const askWaterQuality = async (question) => {
  const response = await fetch('http://localhost:8000/chat', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ question, session_id: sessionId }),
  });
  const data = await response.json();
  return data.answer;
};
```

---

## 🧪 Running Tests

```bash
pytest tests/ -v
pytest tests/ -v --tb=short   # shorter tracebacks
pytest tests/test_services.py::TestTopicGuard -v   # single class
```

---

## 📊 WQI Reference Scale

| WQI Score | Category | Interpretation |
|-----------|----------|----------------|
| 90 – 100 | 🟢 Excellent | Safe for drinking, minimal treatment |
| 70 – 89 | 🟡 Good | Safe with standard treatment |
| 50 – 69 | 🟠 Medium | Suitable for irrigation, treatment needed for drinking |
| 25 – 49 | 🔴 Bad | High pollution, significant treatment required |
| 0 – 24 | ⚫ Very Bad | Severely polluted, not fit for consumption |

---

## 🔮 Phase 2: Scaling to PostgreSQL + React

### PostgreSQL Migration

```python
# Replace CSV loading with SQLAlchemy
from sqlalchemy import create_engine
engine = create_engine(settings.database_url)
df = pd.read_sql("SELECT * FROM water_bodies", engine)
```

### React Dashboard Integration

```
React Dashboard
├── /map           → GET /waterbodies (map markers, WQI colours)
├── /chat          → POST /chat (chatbot widget)
├── /compare       → POST /compare (side-by-side comparison)
├── /trends        → GET /wqi/{location}?start=&end= (time series)
└── /body/{id}     → GET /waterbody/{name} (detail view)
```

### Recommended React Libraries

- **Map**: `react-leaflet` or `deck.gl`
- **Charts**: `recharts` or `Chart.js`
- **Chat widget**: custom component calling `POST /chat`
- **State**: Zustand or Redux Toolkit
- **HTTP**: TanStack Query (react-query)

### CORS is pre-configured for:

- `http://localhost:3000` (Create React App)
- `http://localhost:5173` (Vite)

Add production domain to `CORS_ORIGINS` in `.env`.

---

## 🔑 Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `GROQ_API_KEY` | ✅ | — | Your Groq API key |
| `GROQ_MODEL_NAME` | ❌ | `llama-3.1-8b-instant` | Groq model to use |
| `EMBEDDING_MODEL_NAME` | ❌ | `sentence-transformers/all-MiniLM-L6-v2` | HuggingFace embedding model |
| `FAISS_INDEX_PATH` | ❌ | `./app/data/faiss_index` | Where to save/load FAISS index |
| `DATASET_PATH` | ❌ | `./app/data/water_quality_data.csv` | Path to your CSV dataset |
| `TOP_K_RESULTS` | ❌ | `3` | Documents to retrieve per query |
| `APP_ENV` | ❌ | `development` | `development` or `production` |

---

## 📄 License

MIT License. Built for water quality research and monitoring.
