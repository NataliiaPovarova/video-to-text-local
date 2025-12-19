FROM python:3.11-slim

ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    TORCH_INDEX_URL=${TORCH_INDEX_URL}

WORKDIR /app

RUN apt-get update && \
    apt-get install -y --no-install-recommends ffmpeg && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade pip && \
    if [ -n "$TORCH_INDEX_URL" ]; then \
        pip install --no-cache-dir -r requirements.txt --extra-index-url "$TORCH_INDEX_URL"; \
    else \
        pip install --no-cache-dir -r requirements.txt; \
    fi

COPY . .

ENTRYPOINT ["python", "main.py"]
CMD ["--help"]

