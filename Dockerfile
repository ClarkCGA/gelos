# olmoearth-pretrain requires torch>=2.7,<2.8 — keep the base torch inside that
# range so the requirements install doesn't replace the baked-in torch.
FROM pytorch/pytorch:2.7.1-cuda12.8-cudnn9-runtime AS base

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
# custom_modules/ is auto-imported by terratorch from the working directory
# (WORKDIR=/app) for user-defined per-project backbones.
# OlmoEarth backbone is now inside gelos/backbones/ (copied with gelos/ above).
# models/ contains other per-project model modules (e.g. prithvi_eo_v2.py).
COPY models/ /app/models/
COPY custom_modules/ /app/custom_modules/
RUN uv pip install --system --no-cache --no-deps -e . && \
    chmod -R a+w /app

FROM base AS test

COPY tests/ /app/tests/
RUN chmod -R a+w /app/tests

CMD ["python", "-m", "pytest", "tests"]

FROM base AS prod

CMD ["make", "-h"]
