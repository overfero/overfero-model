import logging
import logging.config
import sys
from io import StringIO
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional, Union

from overfero.utils.io_utils import open_file, write_yaml_file

if TYPE_CHECKING:
    from overfero.config_schemas.config_schema import Config
import hydra
import yaml
from hydra.types import TaskFunction
from omegaconf import DictConfig, OmegaConf

from overfero.config_schemas import config_schema


def get_config(config_path: str, config_name: str) -> TaskFunction:
    setup_config()
    setup_logger()

    def main_decorator(task_function: TaskFunction) -> Any:
        @hydra.main(config_path=config_path, config_name=config_name, version_base=None)
        def decorated_main(dict_config: Optional[DictConfig] = None) -> Any:
            config = OmegaConf.to_object(dict_config)
            return task_function(config)

        return decorated_main

    return main_decorator


def setup_config() -> None:
    config_schema.setup_config()


def setup_logger() -> None:
    with open("./overfero/configs/hydra/job_logging/custom.yaml", "r") as stream:
        config = yaml.load(stream, Loader=yaml.FullLoader)
    logging.config.dictConfig(config)


def save_config_as_yaml(config: Union["Config", DictConfig], save_path: str) -> None:
    text_io = StringIO()
    text_io.writelines(
        [
            f"# Do not edit this file. It is automatically generated by {sys.argv[0]}.\n",
            "# If you want to modify configuration, edit source files in overfero/configs directory.\n",
            "\n",
        ]
    )

    config_header = load_config_header()
    text_io.write(config_header)
    text_io.write("\n")

    OmegaConf.save(config, text_io, resolve=True)
    with open_file(save_path, "w") as f:
        f.write(text_io.getvalue())


def load_config_header() -> str:
    config_header_path = Path("./overfero/configs/automatically_generated/full_config_header.yaml")
    if not config_header_path.exists():
        config_header = {
            "defaults": [
                # {"override hydra/job_logging": "custom"},
                {"override hydra/hydra_logging": "disabled"},
                "_self_",
            ],
            "hydra": {"output_subdir": None, "run": {"dir": "."}},
        }
        config_header_path.parent.mkdir(parents=True, exist_ok=True)
        write_yaml_file(str(config_header_path), config_header)

    with open(config_header_path, "r") as f:
        return f.read()
