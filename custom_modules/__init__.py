"""terratorch custom-module registration for the OlmoEarth backbone.

terratorch auto-imports a top-level ``custom_modules`` package relative to the
working directory at startup (see
``terratorch/registry/custom_registry.py``: it appends ``os.getcwd()`` to
``sys.path`` and ``importlib.import_module("custom_modules")``). GELOS runs with
``WORKDIR=/app`` and ``PYTHONPATH=/app`` in Docker, where this package is copied,
so importing it triggers registration. Run local (non-Docker) invocations from the
repo root so the auto-import finds this package.

Registration mechanism (verified against terratorch's ``Registry``): the
``@registry.register`` decorator keys the registry on ``constructor.__name__``.
We therefore register the factory functions named exactly ``olmoearth_v1_nano``,
``olmoearth_v1_tiny``, ``olmoearth_v1_base``, ``olmoearth_v1_large``, and their
S1+S2 counterparts ``olmoearth_v1_{nano,tiny,base,large}_s1s2`` into
``TERRATORCH_BACKBONE_REGISTRY`` (the default source behind ``BACKBONE_REGISTRY``);
``BACKBONE_REGISTRY.build("olmoearth_v1_base", **model_args)`` then resolves them.

The OlmoEarth wrapper depends on the optional, heavy ``olmoearth-pretrain`` package.
Importing the wrapper module does NOT require it (the dependency is lazy-imported
inside ``__init__``/``forward``), so registration is safe even when the extra is
absent. We still guard the import defensively so a broken/optional install can
never crash terratorch startup for users who don't use OlmoEarth.
"""

import logging

logger = logging.getLogger("terratorch")

# OlmoEarth backbone registration was moved to gelos/backbones/olmoearth_backbone.py
# (imported from gelos/generation.py). This file is reserved for user-defined
# per-project backbones that terratorch auto-imports from the working directory.
