from pathlib import Path
import torch
from gelos.gelosdatamodule import GELOSDataModule
import yaml
from gelos import config
from lightning.pytorch import Trainer
from pathlib import Path
from loguru import logger
from tqdm import tqdm
import typer

app = typer.Typer()
from gelos.config import PROJ_ROOT, PROCESSED_DATA_DIR, DATA_VERSION, RAW_DATA_DIR
from terratorch.tasks import EmbeddingGenerationTask
from lightning.pytorch.cli import instantiate_class

class LenientEmbeddingGenerationTask(EmbeddingGenerationTask):
    def check_file_ids(self, file_ids, x):
        return

def generate_embeddings(yaml_path: Path) -> None:

    with open(yaml_path, "r") as f:
        yaml_config = yaml.safe_load(f)
    
    print(yaml.dump(yaml_config))

    model_name = yaml_config['model']['init_args']['model']
    output_dir = PROCESSED_DATA_DIR / DATA_VERSION / model_name
    output_dir.mkdir(exist_ok=True, parents=True)
    data_root = RAW_DATA_DIR / DATA_VERSION
    marker_file = output_dir / ".embeddings_complete"
    if (marker_file).exists():
        print("embeddings already complete, skipping...")
        return

    # add variables to yaml config so it can be passed to classes
    yaml_config['data']['init_args']['data_root'] = data_root
    yaml_config['model']['init_args']['output_dir'] = output_dir

    # instantiate transform classes if they exist
    if "transform" in yaml_config["data"]["init_args"].keys():
          yaml_config["data"]["init_args"]["transform"] = [
                instantiate_class(args = (), init=class_path) for class_path in yaml_config["data"]["init_args"]["transform"]
          ]

    gelos_datamodule = GELOSDataModule(**yaml_config['data']['init_args'])
    task = LenientEmbeddingGenerationTask(**yaml_config['model']['init_args'])

    device = 'gpu' if torch.cuda.is_available() else 'cpu'
    trainer = Trainer(accelerator=device, devices=1)

    trainer.predict(model=task, datamodule=gelos_datamodule)
    marker_file.touch()
    print("marking embeddings as complete")
    
@app.command()
def main():
    yaml_config_directory = PROJ_ROOT / 'gelos' / 'configs'
    yaml_paths = list(yaml_config_directory.glob('*.yaml'))
    print(f"yamls to process: {yaml_paths}")
    for yaml_path in yaml_paths:
        generate_embeddings(yaml_path)

if __name__ == "__main__":
    app()
