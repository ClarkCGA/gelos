import re
from typing import Any


def resolve_chip_year(dataset, index: int, cfg: dict[str, Any]) -> int:
    """Resolve the calendar year for one chip per a ``year_strategy`` config block.

    Supported strategies:

    - ``from_filename``: regex-extract from a sensor's first-timestep filename.
      Requires ``sensor`` and ``pattern`` keys; the pattern must include one
      capturing group whose value parses as ``int``.
    - ``from_column``: read from a chip-tracker column. Requires the dataset
      class to expose a ``gdf`` attribute (``ExampleGELOSDataSet`` does).
      Requires ``column`` key.

    Raises:
        ValueError: when the strategy is unknown or the resolved value is missing.
    """
    if "type" not in cfg:
        raise ValueError(f"year_strategy missing 'type' key: {cfg!r}")
    strategy = cfg["type"]

    if strategy == "from_filename":
        sensor = cfg["sensor"]
        pattern = cfg["pattern"]
        path = dataset._get_file_paths(index, sensor)[0]
        match = re.search(pattern, path.name)
        if not match:
            raise ValueError(
                f"year_strategy from_filename: pattern '{pattern}' did not match "
                f"'{path.name}' for chip {index}"
            )
        return int(match.group(1))

    if strategy == "from_column":
        column = cfg["column"]
        if not hasattr(dataset, "gdf"):
            raise ValueError(
                "year_strategy from_column requires the dataset class to expose a 'gdf' attribute"
            )
        return int(dataset.gdf[column].iloc[index])

    raise ValueError(f"unknown year_strategy type: '{strategy}'")
