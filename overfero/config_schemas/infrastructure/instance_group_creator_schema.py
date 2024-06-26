from dataclasses import dataclass

from omegaconf import SI

from overfero.config_schemas.infrastructure.instance_template_creator_schema import (
    InstanceTemplateCreatorConfig,
)


@dataclass
class InstanceGroupCreatorConfig:
    _target_: str = "instance_group_creator.InstanceGroupCreator"
    instance_template_creator: InstanceTemplateCreatorConfig = InstanceTemplateCreatorConfig()
    name: str = SI("${infrastructure.mlflow.experiment_name}-${infrastructure.mlflow.run_name}-${now:%Y%m%d%H%M%S}")
    node_count: int = 1
    project_id: str = SI("${infrastructure.project_id}")
    zone: str = SI("${infrastructure.zone}")
