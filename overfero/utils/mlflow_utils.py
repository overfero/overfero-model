import os
from contextlib import contextmanager
from typing import Iterable, Optional

import mlflow

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")


@contextmanager
def activate_mlflow(
    experiment_name: Optional[str] = None, run_id: Optional[str] = None, run_name: Optional[str] = None
) -> Iterable[mlflow.ActiveRun]:
    set_experiment(experiment_name)

    with mlflow.start_run(run_name=run_name, run_id=run_id) as run:
        yield run


def set_experiment(experiment_name: Optional[str] = None) -> None:
    if experiment_name == None:
        experiment_name = "Default"

    try:
        mlflow.create_experiment(experiment_name)
    except mlflow.exceptions.RestException:
        pass
    mlflow.set_experiment(experiment_name)
