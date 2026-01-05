# syntax=docker/dockerfile:1.7

ARG PYTHON_VERSION=3.11-alpine

FROM python:${PYTHON_VERSION} AS base
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=off \
    PIP_DISABLE_PIP_VERSION_CHECK=on \
    PIP_DEFAULT_TIMEOUT=100
RUN addgroup -S app && adduser -S app -G app
WORKDIR /app

FROM base AS builder
RUN apk add --no-cache build-base curl \
    && pip install uv
COPY requirements.txt .
RUN uv pip install --system -r requirements.txt

FROM base AS runtime
COPY --from=builder /usr/local /usr/local
COPY . .
USER app
EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
