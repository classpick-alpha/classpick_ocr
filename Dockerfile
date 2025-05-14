FROM python:3.9-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app


FROM base AS builder

COPY Pipfile Pipfile.lock* /app/

RUN pip install --no-cache-dir pipenv && \
    pipenv install --deploy --system && \
    pip uninstall -y pipenv

RUN python - <<'PY'
from paddleocr import PaddleOCR
PaddleOCR(use_angle_cls=True, lang="korean")
PY


FROM base AS runtime

COPY --from=builder /usr/local/lib/python3.9/site-packages /usr/local/lib/python3.9/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin
COPY --from=builder /root/.paddleocr /root/.paddleocr

COPY . /app

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
