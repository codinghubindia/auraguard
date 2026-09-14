import os
import json
import pytest

MANIFEST_PATHS = [
    os.path.expanduser('~/naviguard_ws/src/naviguard_rellis/config/rellis_manifest.json'),
    os.path.expanduser('~/naviguard_ws/docs/rellis_manifest.json'),
]


def test_manifest_structure_and_required_keys():
    found = False
    for path in MANIFEST_PATHS:
        if os.path.exists(path):
            with open(path, 'r') as f:
                manifest = json.load(f)
            found = True
            break
    assert found, "rellis_manifest.json must exist"

    required_keys = [
        "dataset_name",
        "dataset_version",
        "sequence_id",
        "source_path",
        "files_used",
        "file_sizes",
        "calibration_source",
        "pose_source",
        "pointcloud_source",
        "semantic_source",
        "terrain_generation_method",
        "mesh_generation_method",
        "coordinate_transform",
        "approximation_notes"
    ]
    for key in required_keys:
        assert key in manifest, f"Missing key in manifest: {key}"


def test_manifest_provenance_truth():
    found = False
    for path in MANIFEST_PATHS:
        if os.path.exists(path):
            with open(path, 'r') as f:
                manifest = json.load(f)
            found = True
            break
    assert found

    source_path = os.path.expanduser(manifest["source_path"])
    # If the physical raw dataset directory is not installed, the manifest must truthfully record it
    if not os.path.exists(source_path):
        assert manifest["dataset_status"] == "NOT_INSTALLED_LOCALLY"
        assert manifest["approximation_notes"]["physical_dataset_present"] is False
        assert "SYNTHETIC" in manifest["approximation_notes"]["status"]
        assert manifest["approximation_notes"]["dashboard_label"] == "RELLIS-INSPIRED SYNTHETIC ENVIRONMENT"
    else:
        assert manifest["dataset_status"] in ("OFFICIAL_REPO_INSTALLED", "INSTALLED_LOCALLY")
        assert manifest["approximation_notes"]["physical_dataset_present"] is True


def test_manifest_coordinate_transform():
    found = False
    for path in MANIFEST_PATHS:
        if os.path.exists(path):
            with open(path, 'r') as f:
                manifest = json.load(f)
            found = True
            break
    assert found

    ct = manifest["coordinate_transform"]
    assert "origin" in ct
    assert "resolution" in ct
    assert ct["resolution"] == 0.05
    assert ct["width"] == 600
    assert ct["height"] == 600
    assert "axis_convention" in ct
    assert "REP-103" in ct["axis_convention"]
