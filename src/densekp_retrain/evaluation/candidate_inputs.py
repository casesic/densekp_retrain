"""Freeze candidate sources and references before loading any model."""

from datetime import datetime, timezone
import json
import shutil
from .records import FrozenInputs
from densekp_retrain.io.json import write_json


def prepare_inputs(indices, labels, reference_report, output):
    "Read references and accepted indices in supplied order. The CLI separately validates source isolation, before writing selection or loading a model."
    references = {}
    if reference_report:
        references = {
            c["id"]: c
            for c in json.loads(reference_report.read_text())["cases"]
            if c["batch"] in labels
        }
    sources = []
    partitions = {}
    partition_indices = {}
    for label, path in zip(labels, indices):
        index = json.loads(path.read_text())
        if any((row["quality_state"] != "candidate_accepted" for row in index["rows"])):
            raise ValueError("Inference requires pre-frozen, accepted candidate rows")
        sources.append((label, index["rows"]))
        partitions[label] = index.get("split", "integration_only")
        if partitions[label] in ("train", "val"):
            if partitions[label] in partition_indices:
                raise ValueError("Use one index per partition per baseline run")
            partition_indices[partitions[label]] = index
        shutil.copyfile(path, output / f"{label}_frozen_index.json")
    return FrozenInputs(sources, partitions, partition_indices, references)


def write_selection(inputs, output):
    "Record the unchanged frozen selection policy and batch denominators."
    sources = inputs.sources
    partitions = inputs.partitions
    split_scope = inputs.split_scope
    write_json(
        output / "selection_frozen.json",
        dict(
            utc=datetime.now(timezone.utc).isoformat(),
            rows={label: len(rows) for label, rows in sources},
            selection="input indices frozen before model loading",
            scale_bins="exclusive oracle bbox height in synthetic ERP pixels: small <96, medium 96..191, large >=192",
            fixed_controls="first indexed row in each batch/action/layout category; determined before prediction",
            split=split_scope,
            partitions=partitions,
        ),
    )
