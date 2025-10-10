from shapely import Polygon
import os
import gc
import pandas as pd
import geopandas as gpd
import pytest
from torch.utils.data import DataLoader
from utils import create_dummy_image

@pytest.fixture
def dummy_gelos_data(tmp_path) -> str:
    base_dir = tmp_path / "gelos"
    base_dir.mkdir()
    metadata_filename = "cleaned_df.geojson"
    metadata_path = base_dir / metadata_filename
    
    # Create a GeoDataFrame that matches the GeoJSON structure
    data = {
        "chip_id": [0],
        "sentinel_2_dates": [["20230218", "20230419", "20230713", "20231230"]],
        "sentinel_1_dates": [["20230218", "20230419", "20230712", "20231227"]],
        "landsat_dates": [["20230217", "20230524", "20230921", "20231218"]],
        "land_cover": [2],
    }
    for sentinel_2_dates, chip_id in zip(data['sentinel_2_dates'], data['chip_id']):
        for date in sentinel_2_dates:
            create_dummy_image(base_dir / f"sentinel_2_{chip_id:06}_{date}.tif", (96, 96, 13), range(255))
    for landsat_dates, chip_id in zip(data['landsat_dates'], data['chip_id']):
        for date in landsat_dates:
            create_dummy_image(base_dir / f"landsat_{chip_id:06}_{date}.tif", (96, 96, 7), range(255))
    for sentinel_1_dates, chip_id in zip(data['sentinel_1_dates'], data['chip_id']):
        for date in sentinel_1_dates:
            create_dummy_image(base_dir / f"sentinel_1_{chip_id:06}_{date}.tif", (96, 96, 7), range(255))
    for chip_id in data['chip_id']:
        create_dummy_image(base_dir / f"{chip_id:06}_dem.tif", (96, 96), range(255))

    # Create a dummy polygon geometry
    polygon = Polygon([
        (21.8299, 4.2812), (21.8299, 4.2899), 
        (21.8212, 4.2899), (21.8212, 4.2812), 
        (21.8299, 4.2812)
    ])
    
    gdf = gpd.GeoDataFrame(data, geometry=[polygon], crs="EPSG:4326")
    
    # Save as GeoJSON
    gdf.to_file(metadata_path, driver='GeoJSON')
    
    return str(base_dir)

def test_gelos_datamodule(dummy_gelos_data):
    from gelos.gelosdatamodule import GELOSDataModule
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
    assert "S1" in batch, "Key S1 not found on predict_dataloader"
    assert "S2" in batch, "Key S2 not found on predict_dataloader"

    gc.collect()