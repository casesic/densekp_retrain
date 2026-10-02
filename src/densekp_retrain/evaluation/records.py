"""Data passed between candidate evaluation stages; no model or CLI imports."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class FrozenInputs:
    sources: list
    partitions: dict
    partition_indices: dict
    references: dict

    @property
    def split_scope(self):
        return "source_isolated_initial_holdout" if self.partition_indices else "integration_only"


@dataclass
class LoadedEvaluation:
    model: Any
    device: Any
    checkpoint: Path
    checkpoint_stat: Any
    audit: dict


@dataclass
class SampleCache:
    datasets: dict = field(default_factory=dict)
    scenes: dict = field(default_factory=dict)
    loader_ids: dict = field(default_factory=dict)


@dataclass
class CandidateSample:
    """One Dataset item and its GT, before inference.

    item['img'] is normalized float32 [3,256,256]; target_crop is [138,2]
    in full-square normalized coordinates. visible and each mask are bool [138].
    annotations retains scene arrays in ERP pixels/world metres and person_index
    selects their person axis. No 32-pixel shift is applied here.
    """

    directory: Path
    scene_key: str
    item: dict
    person: dict
    annotations: Any
    person_index: int
    target_crop: Any
    visible: Any
    masks: dict


@dataclass
class ScoredCandidate:
    case: dict
    state: dict
    arrays: dict


@dataclass
class Observations:
    cases: list = field(default_factory=list)
    states: dict = field(default_factory=dict)
    pairs: dict = field(default_factory=dict)
    fixed: dict = field(default_factory=dict)
