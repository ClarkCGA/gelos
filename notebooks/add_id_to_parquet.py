import yaml
from gelos.config import PROCESSED_DATA_DIR, DATA_DIR, DATA_VERSION, RAW_DATA_DIR
import pandas as pd
import geopandas as gpd
from gelos import config
from tqdm import tqdm
from pathlib import Path
yaml_config_directory = config.PROJ_ROOT / 'gelos' / 'configs'
for yaml_filepath in yaml_config_directory.glob("*prithvi*"):
    with open(yaml_filepath, "r") as f:
        yaml_config = yaml.safe_load(f)
    print(yaml.dump(yaml_config))
    model_name = yaml_config['model']['init_args']['model']
    model_title = yaml_config['model']['title']
    output_dir = PROCESSED_DATA_DIR / DATA_VERSION / model_name
    data_root = RAW_DATA_DIR / DATA_VERSION

    embedding_extraction_strategies = yaml_config['embedding_extraction_strategies']
    yaml_config['data']['init_args']['data_root'] = data_root
    yaml_config['model']['init_args']['output_dir'] = output_dir
    embeddings_directories = [item for item in output_dir.iterdir() if item.is_dir()]
    for embeddings_directory in embeddings_directories:
        print(str(embeddings_directory))
        for file in tqdm(sorted(embeddings_directory.glob("*.parquet"))):
            df = pd.read_parquet(file)
            file_id = int(file.stem.split("_")[0])  # 0000 -> 0
            df["file_id"] = file_id
            df.to_parquet(file)

