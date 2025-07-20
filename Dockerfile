# ---- Base image -------------------------------------------------
FROM python:3.11-slim

# ---- Environment hygiene ---------------------------------------
ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=on         \
    PIP_DISABLE_PIP_VERSION_CHECK=on

# ---- System deps (tiny) ----------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential git curl && \
    rm -rf /var/lib/apt/lists/*

# ---- Set workdir -----------------------------------------------
WORKDIR /app

# ---- Python dependencies ---------------------------------------
# Leverage Docker layer cache: copy *just* requirements first
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ---- Copy source code ------------------------------------------
COPY app/ ./app/

# ---- Expose Streamlit port -------------------------------------
EXPOSE 80

# ---- Launch -----------------------------------------------------
# * headless = no browser pop-up
# * PORT var makes it Fly.io / Render friendly
CMD ["streamlit", "run", "app/main.py", "--server.headless=true", "--server.address=0.0.0.0", "--server.port=8501"]

