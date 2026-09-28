"""Loads a dataset profile's queries file. Shared by capture_node_snapshots.py,
test_quality.py, and the scenario-builder modules (parity/scenarios/) --
pulled out of capture_node_snapshots.py so the scenario modules can derive a
real sample query from the active profile without a circular import back
into the capture script.
"""

from __future__ import annotations

import json

from parity.dataset_profile import DatasetProfile


def load_queries(profile: DatasetProfile) -> list[dict]:
    """Normalize either accepted queries_file shape into a uniform
    [{"query_id": str, "query": str}, ...] list. The shape (dict-list vs.
    flat string list) is determined by inspecting the parsed content, not
    the file extension -- both JSON and YAML are legal for either shape per
    the design spec's "Dataset profiles" section, so extension alone cannot
    discriminate."""
    text = profile.queries_file.read_text()
    if profile.queries_file.suffix == ".json":
        raw = json.loads(text)
    else:
        import yaml

        raw = yaml.safe_load(text)

    if raw and isinstance(raw[0], dict):
        return [{"query_id": str(entry["query_id"]), "query": entry["query"]} for entry in raw]
    return [{"query_id": str(i), "query": q} for i, q in enumerate(raw)]
