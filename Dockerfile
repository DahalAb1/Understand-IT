
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=on         \
    PIP_DISABLE_PIP_VERSION_CHECK=on

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential git curl && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY app/ ./app/

EXPOSE 80

#to run without fastAPI: CMD ["streamlit", "run", "app/main.py", "--server.headless=true", "--server.address=0.0.0.0", "--server.port=8501"]
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]

 
