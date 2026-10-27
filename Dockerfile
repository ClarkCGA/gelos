# The base torch is kept at 2.7.1 for terratorch 1.1.1 (and the released
# olmoearth-pretrain 0.1.x, which pins torch>=2.7,<2.8), so the requirements
# install doesn't replace the baked-in torch. The olmoearth-pretrain upstream
# snapshot below is installed without deps for the same reason.
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

# OlmoEarth v1.3 (allenai/OlmoEarth-v1_3-Base) needs olmoearth_pretrain's
# AttentionPoolerConfig, which no PyPI release (<=0.1.2) ships yet. Install the
# upstream main snapshot that has it, --no-deps so its torch>=2.9 pin does not
# replace the baked-in torch 2.7.1 (the inference path GELOS uses runs on 2.7.1).
# Bump the SHA (or switch back to a PyPI version) once a release includes it.
# Keep in sync with _OLMOEARTH_PRETRAIN_V1_3_SNAPSHOT in gelos/backbones/olmoearth_backbone.py.
RUN uv pip install --system --no-cache --no-deps \
    "olmoearth-pretrain @ https://github.com/allenai/olmoearth_pretrain/archive/7353d1b5105a3ad662809a700c98065fc711f4ab.tar.gz"

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
