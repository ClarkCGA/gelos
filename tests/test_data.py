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
def dummy_gelos_data(tmp_path, monkeypatch) -> str:
    data_root = tmp_path / "data_root"
    monkeypatch.setenv("GELOS_DATA_ROOT", str(data_root))
    base_dir = tmp_path / "gelos"
    base_dir.mkdir()
    metadata_filename = "cleaned_df.geojson"
    metadata_path = base_dir / metadata_filename
    
    # Create a GeoDataFrame that matches the GeoJSON structure
    data = {
        "id": [0],
        "S2L2A_dates": ["20230218,20230419,20230713,20231230"],
        "S1RTC_dates": ["20230218,20230419,20230712,20231227"],
        "landsat_dates": ["20230217,20230524,20230921,20231218"],
        "land_cover": [2],
    }
    for S2L2A_dates, id in zip(data['S2L2A_dates'], data['id']):
        for date in S2L2A_dates.split(','):
            create_dummy_image(base_dir / f"S2L2A_{id:06}_{date}.tif", (96, 96, 13), range(255))
    for landsat_dates, id in zip(data['landsat_dates'], data['id']):
        for date in landsat_dates.split(','):
            create_dummy_image(base_dir / f"landsat_{id:06}_{date}.tif", (96, 96, 7), range(255))
    for S1RTC_dates, id in zip(data['S1RTC_dates'], data['id']):
        for date in S1RTC_dates.split(','):
            create_dummy_image(base_dir / f"S1RTC_{id:06}_{date}.tif", (96, 96, 7), range(255))
    for id in data['id']:
        create_dummy_image(base_dir / f"dem_{id:06}.tif", (96, 96), range(255))

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
    assert "S1RTC" in batch['image'], "Key S1 not found on predict_dataloader"
    assert "S2L2A" in batch['image'], "Key S2 not found on predict_dataloader"

    gc.collect()
