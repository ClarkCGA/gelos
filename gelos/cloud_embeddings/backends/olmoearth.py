import numpy as np


class OlmoEarthBackend:
    """Stub backend for Ai2's OlmoEarth foundation model.

    Pins the protocol shape via a second backend. Production implementation
    will submit a job to the OlmoEarth inference service, poll until the
    result COG is available, download it, then clip/reproject identically
    to :class:`AlphaEarthBackend`.
    """

    def __init__(self, **kwargs) -> None:
        self.init_args = kwargs

    def fetch(
        self,
        bbox: tuple[float, float, float, float],
        crs: str,
        year: int,
    ) -> np.ndarray:
        raise NotImplementedError(
            "OlmoEarth backend is not yet implemented; see GELOS issue tracker."
        )
