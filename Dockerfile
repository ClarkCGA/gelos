FROM pytorch/pytorch:2.8.0-cuda12.9-cudnn9-runtime AS base
# FROM python:slim-bookworm AS base
# install system tools + cleanup in a single RUN
# RUN apt-get update && apt-get install -y --no-install-recommends \
#     make \
#     curl \
#     git \
#     && rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get install -y --no-install-recommends \
    make \
    curl \
    git \
    build-essential \
    libsqlite3-dev \
    zlib1g-dev \
    && rm -rf /var/lib/apt/lists/*
    
RUN git clone https://github.com/felt/tippecanoe.git /tmp/tippecanoe && \
    make -C /tmp/tippecanoe -j && \
    make -C /tmp/tippecanoe install && \
    rm -rf /tmp/tippecanoe

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# COPY . /app/
COPY pyproject.toml /app/
WORKDIR /app

RUN uv venv /opt/venv --python /opt/conda/bin/python && \
    uv pip install --python /opt/venv/bin/python -r pyproject.toml
ENV PATH="/opt/venv/bin:$PATH"

FROM base AS test

RUN uv pip install --python /opt/venv/bin/python -r pyproject.toml --extra test

# slim image for production
FROM pytorch/pytorch:2.8.0-cuda12.9-cudnn9-runtime AS production
# requirements for tippecanoe
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsqlite3-dev zlib1g-dev && rm -rf /var/lib/apt/lists/*

COPY --from=base /usr/local/bin/tippecanoe* /usr/local/bin/
COPY --from=base /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
WORKDIR /app
COPY gelos/ ./gelos
