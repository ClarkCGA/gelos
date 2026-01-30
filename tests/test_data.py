from shapely import Polygon
import pdb
import os
import gc
import pandas as pd
import geopandas as gpd
import pytest
from torch.utils.data import DataLoader
from utils import create_dummy_image
from pathlib import Path
@pytest.fixture
def dummy_gelos_data(tmp_path) -> str:
    base_dir = tmp_path / "gelos"
    base_dir.mkdir()
    project_root = tmp_path / "project_root"
    project_root.mkdir()
    metadata_filename = "gelos_chip_tracker.geojson"
    metadata_path = base_dir / metadata_filename
    
    # Create a GeoDataFrame that matches the GeoJSON structure
    data = {
        "id": [0],
        "year": [2023],
        "S2L2A_paths": ["S2L2A_000001_20230218.tif,S2L2A_000001_20230419.tif,S2L2A_000001_20230713.tif,S2L2A_000001_20231230.tif"],
        "S1RTC_paths": ["S1RTC_000001_20230218.tif,S1RTC_000001_20230419.tif,S1RTC_000001_20230712.tif,S1RTC_000001_20231227.tif"],
        "landsat_paths": ["landsat_000001_20230217.tif,landsat_000001_20230524.tif,landsat_000001_20230921.tif,landsat_000001_20231218.tif"],
        "DEM_paths": ["DEM_000001.tif"],
    }
    for paths in data['S2L2A_paths']:
        for path in paths.split(','):
            create_dummy_image(base_dir / path, (96, 96, 13), range(255))
    for paths in data['S1RTC_paths']:
        for path in paths.split(','):
            create_dummy_image(base_dir / path, (96, 96, 2), range(255))
    for paths in data['landsat_paths']:
        for path in paths.split(','):
            create_dummy_image(base_dir / path, (32, 32, 7), range(255))
    for paths in data['DEM_paths']:
        for path in paths.split(','):
            create_dummy_image(base_dir / path, (96, 96), range(255))

    # Create a dummy polygon geometry
    polygon = Polygon([
        (21.8299, 4.2812), (21.8299, 4.2899), 
        (21.8212, 4.2899), (21.8212, 4.2812), 
        (21.8299, 4.2812)
    ])
    
    gdf = gpd.GeoDataFrame(data, geometry=[polygon], crs="EPSG:4326")
    gdf.to_file(metadata_path, driver='GeoJSON')
    
    return str(base_dir)

def test_gelos_datamodule(dummy_gelos_data):
    from gelos.gelosdatamodule import GELOSDataModule
    dummy_gelos_data = Path(dummy_gelos_data)
    batch_size = 1
    num_workers = 0
    # all bands
    datamodule = GELOSDataModule(
        data_root=dummy_gelos_data,
        batch_size=batch_size,
        num_workers=num_workers,
    )
    datamodule.setup("predict")
    predict_loader: DataLoader = datamodule.predict_dataloader()
    batch = next(iter(predict_loader))
    assert "S1RTC" in batch['image'], "Key S1RTC not found on predict_dataloader"
    assert "S2L2A" in batch['image'], "Key S2L2A not found on predict_dataloader"

    gc.collect()
