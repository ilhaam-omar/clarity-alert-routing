from __future__ import annotations

import json
from dataclasses import fields
from pathlib import Path
from typing import Any, Dict

from .models import CapacityScenario
from .state import StateBuilderConfig
from .uncertainty import ActionSetConfig, EnvelopeConfig


def read_json(path: str | Path) -> Dict[str, Any]:
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(source)
    with source.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {source}")
    return value


def _filter_dataclass_kwargs(cls, values: Dict[str, Any]) -> Dict[str, Any]:
    allowed = {field.name for field in fields(cls)}
    unknown = sorted(set(values) - allowed)
    if unknown:
        raise ValueError(f"Unknown {cls.__name__} setting(s): {unknown}")
    return values


def load_state_config(path: str | Path) -> StateBuilderConfig:
    values = read_json(path)
    return StateBuilderConfig(**_filter_dataclass_kwargs(StateBuilderConfig, values))


def load_uncertainty_config(
    path: str | Path,
) -> tuple[list[EnvelopeConfig], ActionSetConfig]:
    values = read_json(path)
    raw = values.get("envelopes")
    if not isinstance(raw, list) or not raw:
        raise ValueError("uncertainty design must contain a non-empty 'envelopes' list")
    envelopes: list[EnvelopeConfig] = []
    for item in raw:
        if not isinstance(item, dict):
            raise ValueError("Each uncertainty envelope must be an object")
        ranges_raw = item.get("ranges")
        if not isinstance(ranges_raw, dict):
            raise ValueError("Each uncertainty envelope requires a ranges object")
        ranges = {
            str(name): (float(bounds[0]), float(bounds[1]))
            for name, bounds in ranges_raw.items()
        }
        envelopes.append(
            EnvelopeConfig(
                name=str(item["name"]),
                n_points=int(item["n_points"]),
                seed=int(item["seed"]),
                regret_quantile=float(item.get("regret_quantile", 0.95)),
                ranges=ranges,
                anchor_rules=tuple(tuple(map(str, r)) for r in item.get("anchor_rules", [])),
            )
        )
    names = [envelope.name for envelope in envelopes]
    if len(names) != len(set(names)):
        raise ValueError("Uncertainty envelope names must be unique")
    action_values = values.get("action_set", {})
    if not isinstance(action_values, dict):
        raise ValueError("action_set must be an object")
    action_config = ActionSetConfig(
        **_filter_dataclass_kwargs(ActionSetConfig, dict(action_values))
    )
    action_config.validate()
    return envelopes, action_config


def load_capacity_scenarios(path: str | Path) -> list[CapacityScenario]:
    values = read_json(path)
    raw = values.get("scenarios")
    if not isinstance(raw, list) or not raw:
        raise ValueError("capacity scenario file must contain a non-empty 'scenarios' list")
    result: list[CapacityScenario] = []
    for item in raw:
        if not isinstance(item, dict):
            raise ValueError("Each capacity scenario must be a JSON object")
        scenario = CapacityScenario(
            **_filter_dataclass_kwargs(CapacityScenario, dict(item))
        )
        scenario.validate()
        result.append(scenario)
    names = [scenario.name for scenario in result]
    if len(names) != len(set(names)):
        raise ValueError("Capacity scenario names must be unique")
    return result
