"""YAML configuration loader with light schema validation."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class ExperimentConfig:
    """Top-level experiment configuration.

    Attributes mirror the YAML schema in ``configs/``.  Unknown keys are kept
    in :attr:`extra` so configs remain forward-compatible.
    """

    name: str
    stage: str  # pretrain | finetune_conservative | finetune_progressive | local_baseline | zero_shot | walk_forward
    seed: int = 42
    data: dict[str, Any] = field(default_factory=dict)
    model: dict[str, Any] = field(default_factory=dict)
    training: dict[str, Any] = field(default_factory=dict)
    evaluation: dict[str, Any] = field(default_factory=dict)
    walk_forward: dict[str, Any] = field(default_factory=dict)
    xai: dict[str, Any] = field(default_factory=dict)
    output: dict[str, Any] = field(default_factory=dict)
    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "ExperimentConfig":
        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)
        known = {"name", "stage", "seed", "data", "model", "training",
                 "evaluation", "walk_forward", "xai", "output"}
        extra = {k: v for k, v in raw.items() if k not in known}
        return cls(
            name=raw["name"],
            stage=raw["stage"],
            seed=raw.get("seed", 42),
            data=raw.get("data", {}),
            model=raw.get("model", {}),
            training=raw.get("training", {}),
            evaluation=raw.get("evaluation", {}),
            walk_forward=raw.get("walk_forward", {}),
            xai=raw.get("xai", {}),
            output=raw.get("output", {}),
            extra=extra,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "stage": self.stage,
            "seed": self.seed,
            "data": self.data,
            "model": self.model,
            "training": self.training,
            "evaluation": self.evaluation,
            "walk_forward": self.walk_forward,
            "xai": self.xai,
            "output": self.output,
            **self.extra,
        }
