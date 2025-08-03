# Understand‑IT
#### A technology that rephrases language to make ideas clearer and more accessible to a broader audience.
---

## What Is Understand-It?

Legal agreements, academic papers, and even everyday instructions are often written in overly complex, jargon-heavy language. But this complexity is rarely necessary. Most information can be conveyed just as effectively using a simpler, more understandable set of words—without losing meaning or precision.

Unnecessary complexity:

- Wastes valuable human hours  
- Makes education less accessible  
- Raises barriers to entry for many people  

**Understand-It** is a technology designed to decode dense, complicated content—such as contracts, disclosures, academic papers, legal documents, or any kind of text—into clear, accessible language. By simplifying communication, it helps address these challenges and contributes to a more informed, inclusive, and empowered society.

---

### Example of Simplification

**Original (Complex):**  *"The lessee shall indemnify and hold harmless the lessor from any and all liabilities, claims, and demands, whether arising in tort or contract, which may result from the lessee’s occupancy or use of the leased premises."*

**Simplified Version:** *"The person renting must protect the owner from any problems, claims, or lawsuits that happen because of their use of the property."*

By making language easier to understand, Understand-It empowers more people to confidently engage with information that might otherwise seem intimidating or confusing.

---

## 2.  Key Features

| Feature | Notes |
|---------|-------|
| **LLM-powered rewrite** | Currently **FLAN-T5-Large** (Gemma-2B upgrade planned). |
| **Three difficulty levels** | *Basic* · *Intermediate* · *Advanced*. |
| **PDF → PDF** | Upload a PDF → download a simplified PDF. |
| **SQLite chunk cache** | Skips the model on repeated chunks (5-10× faster). |
| **FastAPI endpoints** | `/health`, `/metrics`, `/stats` (metrics coming Week 2). |
| **Streamlit UI** | Drag-and-drop interface with live preview. |
| **Docker & CI** | One-command deploy; tests run in GitHub Actions (badge ↑). |

---

 ## 3. Tech Used

| Layer | Tools |
|-------|-------|
| Model | `google/flan-t5-large` via 🤗 Transformers |
| Backend | FastAPI + Uvicorn, PyTorch, SQLite |
| Front-end | Streamlit |
| Observability | Prometheus & Grafana *(Week 2)* |
| DevOps | Docker • docker-compose • GitHub Actions • PyTest |

--- 
## 4. How to run this program 


## 🚀 Run It Locally

```bash
git clone https://github.com/<YOUR_USER>/<YOUR_REPO>.git
cd Understand-IT
```

# 1) Install deps
```python
 -m pip install -r requirements.txt
```
# 2) Start API (health & metrics)
```bash
uvicorn main:app --reload --port 8000
```
#    http://localhost:8000/health  → {"status":"ok"}

# 3) Start the UI in another tab
```bash
streamlit run app/main.py
```
#    http://localhost:8501  -> drop your pdf here. 


### Or with Docker 
```bash
docker build -t understand-it
docker run -p 8000:8000 -p 8501:8501 understand-it
```

--- 
##Tests 
```bash
pytest -q          # local
```

---
# license 

Apache 2.0 © Abhinesh Dahal

