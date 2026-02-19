
# adapted from https://github.com/IBM/terratorch.git
import os
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from PIL import Image
from rasterio.transform import from_origin
from shapely.geometry import Point

def create_dummy_tiff(path: str, shape: tuple, pixel_values: int | list, min_size: int = None) -> None:

    if type(pixel_values) == int:
        pixel_values_ = [pixel_values]
    else:
        pixel_values_ = pixel_values

    if not all([type(i)==int for i in pixel_values_]):
        dtype = np.float32
    else:
        dtype = np.uint8

    os.makedirs(os.path.dirname(path), exist_ok=True)
    if min_size is not None:
        if len(shape) == 3:
            h, w, c = shape
            h = max(h, min_size)
            w = max(w, min_size)
            shape = (h, w, c)
        elif len(shape) == 2:
            h, w = shape
            h = max(h, min_size)
            w = max(w, min_size)
            shape = (h, w)
    if len(shape) == 3:
        h, w, c = shape
        data = np.random.choice(pixel_values, size=(h, w, c), replace=True).astype(dtype)
        data = np.transpose(data, (2, 0, 1))
    elif len(shape) == 2:
        data = np.random.choice(pixel_values, size=shape, replace=True).astype(dtype)
        data = data[np.newaxis, ...]
    transform = from_origin(0, 0, 1, 1)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=data.shape[1],
        width=data.shape[2],
        count=data.shape[0],
        dtype=data.dtype,
        transform=transform
    ) as dst:
        dst.write(data)

def create_dummy_image(path: str, shape: tuple, pixel_values: list[int]) -> None:

    if not all([type(i)==int for i in pixel_values]):
        dtype = np.float32
    else:
        dtype = np.uint8

    ext = os.path.splitext(path)[1].lower()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if ext in [".tif", ".tiff"]:
        create_dummy_tiff(path, shape, pixel_values)
    else:
        if len(shape) == 3:
            data = np.random.choice(pixel_values, size=shape, replace=True).astype(dtype)
            img = Image.fromarray(data)
        elif len(shape) == 2:
            data = np.random.choice(pixel_values, size=shape, replace=True).astype(dtype)
            img = Image.fromarray(data)
        else:
            msg = "Invalid shape"
            raise ValueError(msg)
        img.save(path)


def create_test_geojson(
    data_root: Path,
    n_samples: int,
    sensors: dict[str, int],
    n_timesteps: dict[str, int],
    img_size: int = 96,
) -> None:
    """Create a minimal gelos_chip_tracker.geojson and dummy tiffs for testing.

    Args:
        data_root: Directory to write geojson and tiff files into.
        n_samples: Number of chip samples to create.
        sensors: Dict mapping sensor name to number of bands, e.g. {"S2L2A": 7, "DEM": 1}.
        n_timesteps: Dict mapping sensor name to number of timesteps, e.g. {"S2L2A": 2, "DEM": 1}.
        img_size: Height and width of dummy tiffs.
    """
    records = []
    for i in range(1, n_samples + 1):
        row = {"id": i, "geometry": Point(float(i), float(i))}
        for sensor, n_bands in sensors.items():
            timesteps = n_timesteps.get(sensor, 1)
            paths = []
            for t in range(timesteps):
                filename = f"{sensor}_{str(i).zfill(6)}_{t:02d}.tif"
                tiff_path = data_root / filename
                shape = (img_size, img_size, n_bands) if n_bands > 1 else (img_size, img_size)
                create_dummy_tiff(str(tiff_path), shape, [1, 2, 3, 4, 5])
                paths.append(filename)
            row[f"{sensor.lower()}_paths"] = ",".join(paths)
        records.append(row)

    gdf = gpd.GeoDataFrame(records, crs="EPSG:4326")
    gdf.to_file(data_root / "gelos_chip_tracker.geojson", driver="GeoJSON")
