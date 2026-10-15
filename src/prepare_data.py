"""Prepare immutable WiTra splits without PyTorch or test-fitted transforms.

KNN is a training-only heuristic in normalized feature space. Raw duplicates
are retained and assigned to the same internal split, not declared independent.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from dataprocess import preprocessing as dp

PROTOCOL_VERSION = "witra-revision-v1"


def canonical_json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False)


def metadata_digest(metadata):
    return hashlib.sha256(canonical_json(
        {k: v for k, v in metadata.items() if k != "prepared_hash"}
    ).encode("utf-8")).hexdigest()


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def columns(dataset):
    if dataset == "uji":
        return "LONGITUDE", "LATITUDE", "BUILDINGID", "FLOOR"
    if dataset == "uts":
        return "Pos_x", "Pos_y", "Building_ID", "Floor_ID"
    if dataset == "tampere":
        return "Pos_x", "Pos_y", "Building_ID", "Floor"
    raise ValueError(f"Unknown dataset: {dataset}")


def load_raw(dataset, data_root):
    root = Path(data_root)
    if dataset == "tampere":
        folder = root / "tampere_data"
        train, test, _ = dp.load_tampere(str(folder))
        paths = [folder / "raw" / name for name in (
            "Training_rss.csv", "Training_coordinates.csv",
            "Test_rss.csv", "Test_coordinates.csv")]
    else:
        relative = {
            "uji": ("uji_data/raw/trainingData.csv", "uji_data/raw/validationData.csv"),
            "uts": ("UTSIndoorLoc-dataset/uts_data/raw/UTS_training.csv",
                    "UTSIndoorLoc-dataset/uts_data/raw/UTS_test.csv"),
        }
        paths = [root / p for p in relative[dataset]]
        train, test = [pd.read_csv(p) for p in paths]
    for name, frame in (("train", train), ("test", test)):
        frame = frame.copy()  # Consolidate wide RSSI columns before adding IDs.
        frame["row_id"] = [f"{name}:{i}" for i in range(len(frame))]
        if name == "train":
            train = frame
        else:
            test = frame
    return train, test, {str(p.relative_to(root)): file_hash(p) for p in paths}


def validate_raw(frame, dataset, expected_aps=None):
    aps = dp.get_wap_cols(frame)
    if not aps or len(aps) != len(set(aps)) or frame.columns.duplicated().any():
        raise ValueError("Empty/duplicate AP columns or duplicate column names")
    if expected_aps is not None and aps != list(expected_aps):
        raise ValueError("Held-out raw AP names/order differ from supplied training schema")
    x, y, building, floor = columns(dataset)
    for col in aps + [x, y, floor]:
        values = pd.to_numeric(frame[col], errors="raise").to_numpy(dtype=np.float64)
        if not np.isfinite(values).all():
            raise ValueError(f"Non-finite numeric input in {col}")
    floor_values = frame[floor].to_numpy(dtype=np.float64)
    if not np.equal(floor_values, np.floor(floor_values)).all():
        raise ValueError("Floor labels must be integers")
    if frame[building].isna().any() or frame["row_id"].duplicated().any():
        raise ValueError("Missing building label or duplicate row ID")
    return aps


def raw_group_ids(frame, aps):
    """Hash the complete raw fingerprint; verify equality for every hash reuse."""
    seen = {}
    groups = []
    values = np.ascontiguousarray(frame[aps].to_numpy(dtype="<f8"))
    # Canonicalize -0.0 so numeric equality implies the same fingerprint group.
    values[values == 0] = 0
    for row in values:
        payload = row.tobytes()
        group = hashlib.sha256(payload).hexdigest()
        if group in seen and seen[group] != payload:
            raise RuntimeError("Fingerprint hash collision")
        seen[group] = payload
        groups.append(group)
    return np.asarray(groups)


def split_raw(frame, dataset, split_seed=42):
    from sklearn.model_selection import StratifiedGroupKFold
    aps = validate_raw(frame, dataset)
    _, _, building, floor = columns(dataset)
    strata = frame[building].astype(str) + "/" + frame[floor].astype(str)
    groups = raw_group_ids(frame, aps)
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=split_seed)
    train_idx, val_idx = next(splitter.split(np.zeros(len(frame)), strata, groups))
    if set(groups[train_idx]) & set(groups[val_idx]):
        raise AssertionError("Train/validation fingerprint groups overlap")
    expected = set(strata)
    if set(strata.iloc[train_idx]) != expected or set(strata.iloc[val_idx]) != expected:
        raise ValueError("Fixed fold 0 lacks building/floor coverage; revise split policy explicitly")
    if len(train_idx) + len(val_idx) != len(frame):
        raise AssertionError("Raw rows were lost during split")
    return frame.iloc[train_idx].copy(), frame.iloc[val_idx].copy(), groups, train_idx, val_idx


def fit_state(train, dataset):
    aps = validate_raw(train, dataset)
    retained = [ap for ap in aps if not train[ap].eq(100).all()]
    if not retained:
        raise ValueError("No observed training APs")
    x, y, building, floor = columns(dataset)
    xy = train[[x, y]].to_numpy(dtype=np.float64)
    origin = xy.min(axis=0)
    scale = float(np.ptp(xy, axis=0).max())
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("Training coordinate range must be finite and positive")
    # Preserve raw labels as strings for portable, strict class lookup.
    bld_classes = sorted(train[building].astype(str).unique().tolist())
    flr_classes = sorted(int(v) for v in train[floor].unique())
    return {
        "dataset": dataset, "raw_ap_names": aps, "ap_names": retained,
        "min_x": float(origin[0]), "min_y": float(origin[1]), "max_range": scale,
        "missing_value": 100, "replacement_value": -110,
        "norm_method": "mean_center", "mean_center_rel_unheard": -20.0,
        "mean_center_max_diff": 50.0, "mean_center_empty_mean": -80.0,
        "building_classes": bld_classes,
        "floor_classes": flr_classes,
    }


def transform_frame(frame, state):
    dataset = state["dataset"]
    validate_raw(frame, dataset, state["raw_ap_names"])
    x, y, building, floor = columns(dataset)
    feature_frame = frame[state["ap_names"]].copy()
    feature_frame = dp.replace_missing_values(
        feature_frame, state["missing_value"], state["replacement_value"])
    feature_frame = dp.norm_mean_center_WAP(
        feature_frame, rel_unheard=state["mean_center_rel_unheard"],
        max_diff=state["mean_center_max_diff"], re=state["replacement_value"])
    building_map = {label: i for i, label in enumerate(state["building_classes"])}
    floor_map = {label: i for i, label in enumerate(state["floor_classes"])}
    buildings = frame[building].astype(str).map(building_map)
    floors = frame[floor].map(floor_map)
    if buildings.isna().any() or floors.isna().any():
        raise ValueError("Held-out class absent from fitted training class mapping")
    coords_raw = frame[[x, y]].to_numpy(dtype=np.float64)
    coords = (coords_raw - [state["min_x"], state["min_y"]]) / state["max_range"]
    arrays = {
        "features": feature_frame.to_numpy(dtype=np.float32),
        "coords": coords.astype(np.float32), "coords_raw": coords_raw,
        "building": buildings.to_numpy(dtype=np.int64),
        "floor": floors.to_numpy(dtype=np.int64),
        "row_id": frame["row_id"].to_numpy(dtype=str),
    }
    assert_arrays(arrays)
    return arrays


def assert_arrays(arrays):
    n = len(arrays["row_id"])
    if len(set(arrays["row_id"])) != n:
        raise AssertionError("Duplicate prepared row IDs")
    for key, value in arrays.items():
        if len(value) != n:
            raise AssertionError(f"Array length mismatch: {key}")
        if key != "row_id" and not np.isfinite(value).all():
            raise ValueError(f"Non-finite prepared array: {key}")


def augment_training(arrays, dataset, n_aug, k_neighbors=5, seed=42):
    """Original feature-space mixing rule with explicit ancestry and self exclusion."""
    if n_aug < 0 or k_neighbors < 1:
        raise ValueError("aug_n must be nonnegative; aug_k must be positive")
    if not n_aug:
        return arrays, []
    if dataset != "tampere":
        raise ValueError("KNN augmentation is supported only for TUT in this protocol")
    rng = np.random.default_rng(seed)
    rows = {key: [] for key in arrays}
    ancestry = []
    groups = sorted(set(zip(arrays["building"].tolist(), arrays["floor"].tolist())))
    for bld, flr in groups:
        indices = np.flatnonzero((arrays["building"] == bld) & (arrays["floor"] == flr))
        if len(indices) < 2:
            continue
        xy = arrays["coords_raw"][indices]
        for local, anchor in enumerate(indices):
            distance = np.sum((xy - xy[local]) ** 2, axis=1)
            # Stable ties follow original row order; remove SELF by identity.
            order = np.argsort(distance, kind="stable")
            candidates = indices[order[order != local][:k_neighbors]]
            for _ in range(n_aug):
                neighbor = int(rng.choice(candidates))
                alpha = float(rng.uniform(0.1, 0.9))
                a, b = arrays["features"][[anchor, neighbor]]
                both = (a > 0) & (b > 0)
                features = np.where(both, alpha * a + (1 - alpha) * b,
                                    np.maximum(a, b)).astype(np.float32)
                synthetic_id = f"synthetic:{dataset}:{len(ancestry)}"
                item = {
                    "features": features,
                    "coords": (alpha * arrays["coords"][anchor]
                               + (1 - alpha) * arrays["coords"][neighbor]).astype(np.float32),
                    "coords_raw": alpha * arrays["coords_raw"][anchor]
                                  + (1 - alpha) * arrays["coords_raw"][neighbor],
                    "building": arrays["building"][anchor],
                    "floor": arrays["floor"][anchor], "row_id": synthetic_id,
                }
                for key, value in item.items():
                    rows[key].append(value)
                ancestry.append({"row_id": synthetic_id,
                                 "anchor_id": str(arrays["row_id"][anchor]),
                                 "neighbor_id": str(arrays["row_id"][neighbor]),
                                 "alpha": alpha})
    if ancestry:
        arrays = {key: np.concatenate((value, np.asarray(rows[key], dtype=(str if key == "row_id" else value.dtype))))
                  for key, value in arrays.items()}
    assert_arrays(arrays)
    return arrays, ancestry


def prepare_data(dataset, data_root, output_dir, split_seed=42, aug_n=None,
                 aug_k=5, aug_seed=42):
    dataset = "tampere" if dataset.lower() in ("tut", "tampere") else dataset.lower()
    columns(dataset)
    destination = Path(output_dir)
    if destination.exists() and (not destination.is_dir() or any(destination.iterdir())):
        raise FileExistsError(f"Output must be absent or empty: {destination}")
    n_aug = (8 if dataset == "tampere" else 0) if aug_n is None else int(aug_n)
    raw_train, raw_test, source_hashes = load_raw(dataset, data_root)
    train_frame, val_frame, groups, train_idx, val_idx = split_raw(raw_train, dataset, split_seed)
    state = fit_state(train_frame, dataset)
    train_original = transform_frame(train_frame, state)
    val = transform_frame(val_frame, state)
    test = transform_frame(raw_test, state)
    train, ancestry = augment_training(train_original, dataset, n_aug, aug_k, aug_seed)
    train_ids, val_ids = set(train_original["row_id"]), set(val["row_id"])
    if train_ids & val_ids or len(train_ids | val_ids) != len(raw_train):
        raise AssertionError("Raw split IDs overlap or records were lost")
    if any(a["anchor_id"] not in train_ids or a["neighbor_id"] not in train_ids
           or a["anchor_id"] == a["neighbor_id"] for a in ancestry):
        raise AssertionError("Invalid augmentation ancestry")
    prepared = {"train": train, "val": val, "test": test}
    metadata = {
        "protocol_version": PROTOCOL_VERSION, "dataset": dataset,
        "num_features": len(state["ap_names"]),
        "num_buildings": len(state["building_classes"]),
        "num_floors": len(state["floor_classes"]),
        "active_tasks": ["coord"] + (["building"] if len(state["building_classes"]) > 1 else []) + ["floor"],
        "transform": state, "source_hashes": source_hashes,
        "code_hashes": {"prepare_data.py": file_hash(__file__),
                        "dataprocess/preprocessing.py": file_hash(dp.__file__)},
        "config": {"split_seed": split_seed, "splitter": "StratifiedGroupKFold",
                   "n_splits": 5, "validation_fold": 0, "retain_raw_duplicates": True,
                   "grouping": "complete_raw_rssi_sha256_with_collision_check",
                   "norm_method": "mean_center", "aug_n": n_aug, "aug_k": aug_k,
                   "aug_seed": aug_seed, "knn_space": "normalized_features",
                   "knn_single_positive_rule": "max(anchor,neighbor)",
                   "knn_distance": "2D coordinate distance within building/floor",
                   "knn_tie_rule": "stable original row order; exclude self by identity",
                   "online_augmentation": "configured by training runner"},
        "counts": {"raw_train": len(raw_train), "raw_test": len(raw_test),
                   "train_original": len(train_original["row_id"]),
                   "train_synthetic": len(ancestry),
                   **{name: len(arrays["row_id"]) for name, arrays in prepared.items()}},
        "splits": {"train_ids": train_original["row_id"].tolist(),
                   "val_ids": val["row_id"].tolist(), "test_ids": test["row_id"].tolist(),
                   "train_groups": sorted(set(groups[train_idx])),
                   "val_groups": sorted(set(groups[val_idx]))},
        "support": {name: {"outside_unit_square": int(np.any(
            (arrays["coords_raw"] < np.array([state["min_x"], state["min_y"]])) |
            (arrays["coords_raw"] > np.array([state["min_x"], state["min_y"]]) + state["max_range"]), axis=1).sum())}
                    for name, arrays in prepared.items()},
        "ancestry": ancestry, "artifacts": {},
    }
    destination.mkdir(parents=True, exist_ok=True)
    for name, arrays in prepared.items():
        filename = f"{name}.npz"
        np.savez_compressed(destination / filename, **arrays)
        metadata["artifacts"][filename] = file_hash(destination / filename)
    metadata["prepared_hash"] = metadata_digest(metadata)
    (destination / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["uji", "uts", "tampere", "tut"], required=True)
    parser.add_argument("--data-root", required=True, help="Directory containing uji_data, UTSIndoorLoc-dataset, tampere_data")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--aug-seed", type=int, default=42)
    parser.add_argument("--aug-n", type=int)
    parser.add_argument("--aug-k", type=int, default=5)
    args = parser.parse_args()
    metadata = prepare_data(**vars(args))
    print(json.dumps({"marker": "PREPARE_OK", "dataset": metadata["dataset"],
                      "counts": metadata["counts"], "prepared_hash": metadata["prepared_hash"]}))


if __name__ == "__main__":
    main()
