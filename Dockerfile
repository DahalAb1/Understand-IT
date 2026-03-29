FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["streamlit", "run", "app/main.py", "--server.address=0.0.0.0", "--server.port=8000"]

# you'd have to build it first. 
# run the follow in terminal: 
# 
# docker run -p 8000:8000 