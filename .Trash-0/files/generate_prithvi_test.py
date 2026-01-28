
import pandas as pd
from pathlib import Path
import geopandas as gpd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from gelos.gelosdataset import GELOSDataSet
from gelos.gelosdatamodule import GELOSDataModule
import yaml
from gelos import config
from lightning.pytorch import Trainer

yaml_config_directory = config.PROJ_ROOT / 'gelos' / 'configs'
yaml_name = 'prithvi_eo_300m_embedding_generation.yaml'
with open(yaml_config_directory / yaml_name, "r") as f:
    yaml_config = yaml.safe_load(f)
print(yaml.dump(yaml_config))
model_name = yaml_config['model']['init_args']['model']
output_dir = config.INTERIM_DATA_DIR / config.DATA_VERSION / model_name
output_dir.mkdir(exist_ok=True)

data_root = config.RAW_DATA_DIR / config.DATA_VERSION

# add variables to yaml config so it can be passed to classes
yaml_config['data']['init_args']['data_root'] = data_root
yaml_config['model']['init_args']['output_dir'] = output_dir
gelos_datamodule = GELOSDataModule(**yaml_config['data']['init_args'])
from terratorch.tasks import EmbeddingGenerationTask
class LenientEmbeddingGenerationTask(EmbeddingGenerationTask):
    def check_file_ids(self, file_ids, x):
        return
task = LenientEmbeddingGenerationTask(**yaml_config['model']['init_args'])
device = 'gpu' if torch.cuda.is_available() else 'cpu'
# device = 'cpu'
trainer = Trainer(accelerator=device, devices=1)
breakpoint()
trainer.predict(model=task, datamodule=gelos_datamodule)