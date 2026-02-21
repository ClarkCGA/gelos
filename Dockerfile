FROM pytorch/pytorch:2.8.0-cuda12.9-cudnn9-runtime AS base

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    make \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
ENV PYTHONPATH=/app

COPY requirements.txt /app/
RUN uv pip install --system --no-cache -r requirements.txt

COPY pyproject.toml README.md Makefile LICENSE /app/
COPY gelos/ /app/gelos/
RUN uv pip install --system --no-cache --no-deps -e . && \
    chmod -R a+w /app

FROM base AS test

COPY tests/ /app/tests/
RUN chmod -R a+w /app/tests

CMD ["python", "-m", "pytest", "tests"]

FROM base AS prod

CMD ["make", "-h"]
