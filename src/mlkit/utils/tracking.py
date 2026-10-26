import dataclasses
from collections.abc import Sequence
from pathlib import Path
from typing import cast

import mlflow


def setup_mlflow(root: Path, experiment: str) -> None:
    mlflow.set_tracking_uri(f"sqlite:///{(root / 'mlflow.db').resolve().as_posix()}")

    if mlflow.get_experiment_by_name(experiment) is None:
        artifacts = (root / "mlartifacts" / experiment).resolve().as_uri()
        mlflow.create_experiment(experiment, artifact_location=artifacts)

    mlflow.set_experiment(experiment)


def flatten_params(config: object) -> dict[str, str]:
    if not dataclasses.is_dataclass(config) or isinstance(config, type):
        raise TypeError(f"flatten_params expects a dataclass instance, got {type(config).__name__}")

    params: dict[str, str] = {}
    _flatten(dataclasses.asdict(config), "", params)
    return params


def _flatten(value: object, prefix: str, params: dict[str, str]) -> None:
    if isinstance(value, dict):
        for key, item in cast(dict[str, object], value).items():
            _flatten(item, f"{prefix}{key}.", params)
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(cast(Sequence[object], value)):
            _flatten(item, f"{prefix}{index}.", params)
    else:
        params[prefix.removesuffix(".")] = str(value)
