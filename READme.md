

# Understand-IT
**A technology that rephrases language to make ideas clearer and more accessible to a broader audience.**

---

## What Is Understand-IT?

Legal agreements, academic papers, and even everyday instructions are often written in overly complex, jargon-heavy language.  
Most information can be conveyed just as effectively using simpler words—without losing meaning or precision.

Unnecessary complexity:

- Wastes valuable hours  
- Makes education less accessible  
- Raises barriers to entry for many people  

**Understand-IT** decodes dense content—contracts, disclosures, academic papers, legal docs—into clear, accessible language.

---

### Example of Simplification

| Original (complex) | Simplified |
|--------------------|------------|
| “The lessee shall indemnify and hold harmless the lessor from any and all liabilities, claims, and demands, whether arising in tort or contract, which may result from the lessee’s occupancy or use of the leased premises.” | “The person renting must protect the owner from any problems, claims, or lawsuits that happen because of their use of the property.” |

---

## 🔍 Key Features

| Feature | Notes |
|---------|-------|
| **LLM-powered rewrite** | Currently **FLAN-T5-Large** (Gemma-2B upgrade planned) |
| **Three difficulty levels** | *Basic* · *Intermediate* · *Advanced* |
| **PDF → PDF** | Upload a PDF → download a simplified PDF |
| **SQLite chunk cache** | Skips the model on repeated chunks (5-10× faster) |
| **FastAPI endpoints** | `/health`, `/metrics`, `/stats` *(metrics Week 2)* |
| **Streamlit UI** | Drag-and-drop interface with live preview |
| **Docker & CI** | One-command deploy; tests run in GitHub Actions (badge ↑) |

---

## 🛠 Tech Stack

| Layer | Tools |
|-------|-------|
| Model | `google/flan-t5-large` via 🤗 Transformers |
| Backend | FastAPI · Uvicorn · PyTorch · SQLite |
| Front-end | Streamlit |
| Observability | Prometheus & Grafana *(Week 2)* |
| DevOps | Docker · docker-compose · GitHub Actions · PyTest |

---

## 🚀 Run Locally

```bash
git clone https://github.com/<YOUR_USER>/<YOUR_REPO>.git
cd Understand-IT

# 1. install deps
pip install -r requirements.txt

# 2. start API (health & metrics)
uvicorn main:app --reload --port 8000
# http://localhost:8000/health  → {"status":"ok"}

# 3. start the UI (new shell)
streamlit run app/main.py
# http://localhost:8501  → drag-and-drop PDF
```

or with Docker 

```bash
docker build -t understand-it .
docker run -p 8000:8000 -p 8501:8501 understand-it
```

--- 
Tests 
 ```bash
pytest -q
```
--- 
📜 License

Apache 2.0 © Abhinesh Dahal
