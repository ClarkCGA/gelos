from gelos.config import PROCESSED_DATA_DIR, DATA_VERSION, GELOS_BUCKET
from gelos.config import AWS_ACCESS_KEY, AWS_SECRET_KEY, AWS_REGION
import boto3
output_dir = PROCESSED_DATA_DIR / DATA_VERSION
bucket_name = GELOS_BUCKET
bucket_name = "gelos-fm"
s3_client = boto3.client(
    "s3",
    aws_access_key_id=AWS_ACCESS_KEY,
    aws_secret_access_key=AWS_SECRET_KEY,
    region_name=AWS_REGION
)

centroids_pmtiles = output_dir / "centroids.pmtiles"
chips_pmtiles = output_dir / "gelos_chip_tracker.pmtiles"

s3_client.upload_file(str(centroids_pmtiles), bucket_name, f"pmtiles/{centroids_pmtiles.name}")
s3_client.upload_file(str(chips_pmtiles), bucket_name, f"pmtiles/{chips_pmtiles.name}")