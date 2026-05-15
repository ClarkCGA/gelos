from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np


@dataclass(frozen=True)
class FetchRequest:
    """One bbox/year fetch unit; the batchable counterpart of :meth:`CloudEmbeddingBackend.fetch`."""

    bbox: tuple[float, float, float, float]
    crs: str
    year: int
    out_shape: tuple[int, int] | None = None


@runtime_checkable
class CloudEmbeddingBackend(Protocol):
    """Pluggable backend that returns a footprint-clipped embedding raster.

    Backends are stateless from the CLI's perspective: instantiate once,
    call :meth:`fetch` per footprint. Implementations may cache internally
    (e.g. an index parquet, COG handles); the protocol does not prescribe.

    Backends that can amortize I/O across requests (e.g. by grouping into
    one Zarr-chunk read) may implement :meth:`fetch_batch`; the runner
    detects it via ``hasattr`` and falls back to per-request ``fetch``.
    """

    def fetch(
        self,
        bbox: tuple[float, float, float, float],
        crs: str,
        year: int,
        out_shape: tuple[int, int] | None = None,
    ) -> np.ndarray:
        """Return ``[H, W, C]`` embedding pixels covering ``bbox`` in ``crs`` for ``year``.

        Args:
            bbox: ``(minx, miny, maxx, maxy)`` in the chip's ``crs``.
            crs: Authority-formatted CRS string (e.g. ``"EPSG:32618"``).
            year: Calendar year selector for the backend's annual cadence.
            out_shape: ``(H, W)`` of the returned raster; when set, the backend
                resamples its output to exactly that shape so per-patch vectors
                stack cleanly under flatten pooling. When ``None`` (default),
                the backend returns its natural reprojection shape, matching
                the original contract.
        """
        ...

    def fetch_batch(self, requests: list[FetchRequest]) -> list[np.ndarray]:
        """Return one ``[H, W, C]`` array per request in input order.

        Optional. Implementations may group requests to amortize backing-store
        I/O (e.g. one Zarr-chunk read per group). When not implemented, the
        runner falls back to per-request :meth:`fetch` calls.
        """
        ...
