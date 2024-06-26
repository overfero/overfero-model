from dataclasses import dataclass
from typing import Optional

from omegaconf import SI

from overfero.config_schemas.infrastructure.instance_group_creator_schema import (
    InstanceGroupCreatorConfig,
)


@dataclass
class MLFlowConfig:
    mlflow_external_tracking_uri: str = SI("${oc.env:MLFLOW_TRACKING_URI,localhost:6101}")
    mlflow_internal_tracking_uri: str = SI("${oc.env:MLFLOW_INTERNAL_TRACKING_URI,localhost:6101}")
    experiment_name: str = "Default"
    run_name: Optional[str] = None
    run_id: Optional[str] = None
    experiment_id: Optional[str] = None
    experiment_url: str = SI("${.mlflow_external_tracking_uri}/#/experiment/${.experiment_id}/runs/${.run_id}")
    artifact_uri: Optional[str] = None


@dataclass
class InfrastructureConfig:
    project_id: str = "overfero"
    zone: str = "asia-southeast2-a"
    instance_group_creator: InstanceGroupCreatorConfig = InstanceGroupCreatorConfig()
    mlflow: MLFlowConfig = MLFlowConfig()
