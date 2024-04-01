from pathlib import Path
from typing import TYPE_CHECKING

import mlflow

from overfero.config_schemas.config_schema import Config

# from omegaconf import OmegaConf
from overfero.utils.config_utils import get_config, save_config_as_yaml
from overfero.utils.mlflow_utils import activate_mlflow


@get_config(config_path="../configs", config_name="config")
def generate_final_config(config: Config) -> None:
    with activate_mlflow(
        config.infrastructure.mlflow.experiment_name,
        run_id=config.infrastructure.mlflow.run_id,
        run_name=config.infrastructure.mlflow.run_name,
    ) as run:
        run_id: str = run.info.run_id
        experiment_id: str = run.info.experiment_id
        artifact_uri: str = run.info.artifact_uri

        config.infrastructure.mlflow.artifact_uri = artifact_uri
        config.infrastructure.mlflow.run_id = run_id
        config.infrastructure.mlflow.experiment_id = experiment_id

        config_save_dir = Path("overfero/configs/automatically_generated/")
        config_save_dir.mkdir(parents=True, exist_ok=True)
        (config_save_dir / "__init__.py").touch(exist_ok=True)

        yaml_config_save_path = config_save_dir / "config.yaml"
        save_config_as_yaml(config, str(yaml_config_save_path))
        mlflow.log_artifact(str(yaml_config_save_path))


if __name__ == "__main__":
    generate_final_config()
