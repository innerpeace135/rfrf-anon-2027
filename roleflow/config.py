from __future__ import annotations

import dataclasses
import typing
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import yaml

ROOT = Path(__file__).resolve().parent.parent


@dataclass
class ModelConfig:
    model: str
    api_base: str = ""
    api_key: str = "EMPTY"
    concurrency: int = 8


@dataclass
class LibraryConfig:
    seed: str


@dataclass
class SamplingConfig:
    chains: int = 1


@dataclass
class SECPOConfig:
    gr_backbone: str
    group_size: int
    tasks_per_training_step: int
    learning_rate: float
    kl_coefficient: float
    clip_ratio: float
    commit_threshold: float
    edit_advantage_weight: float
    gpus: int
    score_weight: float = 0.25


@dataclass
class WebInteractionConfig:
    environment: str = ""


@dataclass
class DataConfig:
    root: str = "data"
    output_dir: str = "outputs"


@dataclass
class Config:
    domain: str
    benchmark: str
    maximum_cycles: int
    executor: ModelConfig
    global_router: ModelConfig
    library: LibraryConfig
    sampling: SamplingConfig
    data: DataConfig
    judge: Optional[ModelConfig] = None
    secpo: Optional[SECPOConfig] = None
    web_interaction: Optional[WebInteractionConfig] = None

    def output(self, *parts: str) -> Path:
        path = resolve(self.data.output_dir).joinpath(*parts)
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def folder(self, *parts: str) -> Path:
        path = resolve(self.data.output_dir).joinpath(*parts)
        path.mkdir(parents=True, exist_ok=True)
        return path

    def require(self, block: str):
        value = getattr(self, block)
        if value is None:
            raise ValueError(f"the config needs a {block} block")
        return value


def nested_dataclass(hint):
    if dataclasses.is_dataclass(hint):
        return hint
    if typing.get_origin(hint) is typing.Union:
        inner = [argument for argument in typing.get_args(hint) if dataclasses.is_dataclass(argument)]
        return inner[0] if inner else None
    return None


def build(kind, values):
    values = {} if values is None else values
    if not isinstance(values, dict):
        raise ValueError(f"{kind.__name__} must be a mapping")
    hints = typing.get_type_hints(kind)
    unknown = set(values) - set(hints)
    if unknown:
        raise ValueError(f"unknown {kind.__name__} keys: {sorted(unknown)}")
    arguments = {}
    for spec in dataclasses.fields(kind):
        if spec.name in values:
            nested = nested_dataclass(hints[spec.name])
            arguments[spec.name] = build(nested, values[spec.name]) if nested else values[spec.name]
    missing = [
        spec.name
        for spec in dataclasses.fields(kind)
        if spec.name not in arguments and spec.default is dataclasses.MISSING and spec.default_factory is dataclasses.MISSING
    ]
    if missing:
        raise ValueError(f"{kind.__name__} needs {', '.join(missing)}")
    return kind(**arguments)


def resolve(path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def load_config(path) -> Config:
    with open(resolve(path), encoding="utf-8") as handle:
        return build(Config, yaml.safe_load(handle))
