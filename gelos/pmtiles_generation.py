import geopandas as gpd
import subprocess
from gelos.config import RAW_DATA_DIR, PROCESSED_DATA_DIR, DATA_VERSION
import shutil

output_dir = PROCESSED_DATA_DIR / DATA_VERSION
embedding_csv_paths = output_dir.rglob("*.csv")
data_root = RAW_DATA_DIR / DATA_VERSION
chip_gdf = gpd.read_file(data_root / 'gelos_chip_tracker.geojson')
chip_gdf = chip_gdf[[
    "category",
    "sentinel_1_dates",
    "sentinel_2_dates",
    "landsat_dates",
    "id",
    "lat",
    "lon",
    "color",
    "landsat_thumbs",
    "sentinel_1_thumbs",
    "sentinel_2_thumbs",
    "geometry"
    ]]
for embedding_csv_path in embedding_csv_paths:
    embed_df = gpd.read_file(embedding_csv_path)
    embed_df.to_csv(output_dir / embedding_csv_path.name, index=False)
    
print(f"chip tracker columns: {chip_gdf.columns}")
chip_gdf.to_file(output_dir / "gelos_chip_tracker_with_tsne.geojson")
chip_gdf_centroids = chip_gdf.copy()
chip_gdf_centroids["geometry"] = chip_gdf_centroids.geometry.centroid
chip_gdf_centroids = chip_gdf_centroids.set_geometry("geometry")
chip_gdf_centroids.to_file(output_dir / "gelos_centroids_with_tsne.geojson")
chip_gdf_centroids.to_file(output_dir / "points.json")



cmd = f"""
tippecanoe -f -Z5 -z14 \
  -ps \
  --no-tiny-polygon-reduction \
  --no-tile-size-limit \
  --no-feature-limit \
  --drop-densest-as-needed \
  --extend-zooms-if-still-dropping \
  -l gelos_centroids \
  -o {str(output_dir / "centroids.pmtiles")} \
  {str(output_dir / "gelos_centroids_with_tsne.geojson")}
"""

subprocess.run(cmd, shell=True, check=True)

cmd = f"""
tippecanoe -f -Z5 -z14 \
  -ps \
  --no-tiny-polygon-reduction \
  --no-tile-size-limit \
  --no-feature-limit \
  --drop-densest-as-needed \
  --extend-zooms-if-still-dropping \
  -l gelos_chips \
  -o {str(output_dir / "gelos_chip_tracker.pmtiles")} \
  {str(output_dir / "gelos_chip_tracker_with_tsne.geojson")}
"""


subprocess.run(cmd, shell=True, check=True)