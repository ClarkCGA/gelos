FROM pytorch/pytorch:2.8.0-cuda12.9-cudnn9-runtime AS base
# FROM python:slim-bookworm AS base
# install system tools + cleanup in a single RUN
RUN apt-get update && apt-get install -y --no-install-recommends \
    make \
    curl \
    && rm -rf /var/lib/apt/lists/*
    
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# COPY . /app/
COPY pyproject.toml /app/
WORKDIR /app

RUN uv venv /opt/venv --python /opt/conda/bin/python && \
    uv pip install --python /opt/venv/bin/python -r pyproject.toml
ENV PATH="/opt/venv/bin:$PATH"

FROM base AS test

RUN uv pip install --python /opt/venv/bin/python -r pyproject.toml --extra test

# FROM python:slim-bookworm AS production
FROM pytorch/pytorch:2.8.0-cuda12.9-cudnn9-runtime AS production
 
WORKDIR /app

COPY --from=base /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY gelos/ ./gelos
