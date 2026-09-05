FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1
ENV PORT=8080

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY backend/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -U pip setuptools wheel \
    && pip install --no-cache-dir -r /app/requirements.txt

COPY backend/ /app/backend/
COPY frontend/dist/ /app/frontend/dist/
COPY examples/ /app/examples/

ENV PYTHONPATH=/app

EXPOSE 8080

CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8080"]
