"""Leakage-safe WiTra revision runner. Training never opens the official test NPZ.

Train, evaluate, and aggregate are deliberately separate commands. Existing run
directories are never reused; a failed run keeps its manifest and error record.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import random
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, get_worker_info

from model.witra import Witra
from uncertainty_weighting import BACKEND, UPSTREAM_COMMIT, UW_SOURCE, create_weighting, validate_checkpoint_weighting
from floor_protection import FloorUpdateProtection



PROTOCOL = "witra-direct-v1"
DATA_PROTOCOL = "witra-revision-v1"
ROOT = Path(__file__).resolve().parent
VARIANT = "MAIN"  # the complete WiTra configuration reported in the paper
METRIC_NAMES = ("mean_error_m", "median_error_m", "p90_error_m", "building_accuracy_pct", "floor_accuracy_pct")


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def digest_json(value):
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def read_json(path):
    with Path(path).open(encoding="utf-8") as stream:
        return json.load(stream)


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
    os.replace(temporary, path)


def source_hashes():
    names = ["model/witra.py", "dataprocess/preprocessing.py", "prepare_data.py", "trainer.py", "run.py", "uncertainty_weighting.py", "floor_protection.py"]
    names += [path.relative_to(ROOT).as_posix() for path in sorted((ROOT / "LibMTL" / "LibMTL").rglob("*.py"))]
    return {name: sha256_file(ROOT / name) for name in names}


def load_metadata(prepared):
    metadata = read_json(Path(prepared) / "metadata.json")
    if metadata.get("protocol_version") != DATA_PROTOCOL:
        raise ValueError("Prepared data belongs to a different protocol version")
    payload = {key: value for key, value in metadata.items() if key != "prepared_hash"}
    if metadata.get("prepared_hash") != digest_json(payload):
        raise ValueError("Prepared metadata digest does not match its contents")
    for key in ("dataset", "num_features", "num_buildings", "num_floors", "active_tasks", "transform", "artifacts"):
        if key not in metadata:
            raise ValueError(f"Prepared metadata missing {key}")
    active = metadata["active_tasks"]
    expected = ["coord"] + (["building"] if metadata["num_buildings"] > 1 else []) + ["floor"]
    if active != expected:
        raise ValueError(f"Unexpected active tasks: {active}; expected {expected}")
    if metadata["num_floors"] < 2:
        raise ValueError("This protocol expects a nonconstant floor task")
    return metadata


def verify_artifact(prepared, metadata, name):
    # Callers explicitly choose partitions. Training never calls this for test.npz.
    expected = metadata["artifacts"].get(name)
    if not isinstance(expected, str) or sha256_file(Path(prepared) / name) != expected:
        raise ValueError(f"Prepared artifact hash mismatch: {name}")


def seed_everything(seed):
    if seed < 0 or seed >= 2**32:
        raise ValueError("Seed must be in [0, 2**32)")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Fail if an operation lacks a deterministic implementation, rather than
    # silently promising determinism. Device/library changes can still differ.
    torch.use_deterministic_algorithms(True)


def resolve_config(args, metadata):
    values = vars(args) if isinstance(args, argparse.Namespace) else dict(args)
    dataset = metadata["dataset"]
    if dataset not in ("uji", "uts", "tampere"):
        raise ValueError(f"Unknown prepared dataset: {dataset}")
    config = {
        "protocol": PROTOCOL, "dataset": dataset, "variant": VARIANT,
        "seed": int(values.get("seed", 42)), "epochs": int(values.get("epochs", 300)),
        # Paper settings: UJIIndoorLoc 8e-4/32, UTS 1e-4/32, TUT 1e-4/16.
        "batch_size": int({"uji": 32, "uts": 32, "tampere": 16}[dataset] if values.get("batch_size") is None else values["batch_size"]), "num_workers": int(values.get("num_workers", 0)),
        "lr": float({"uji": 8e-4, "uts": 1e-4, "tampere": 1e-4}[dataset]
                    if values.get("lr") is None else values["lr"]),
        "weight_decay": float(values.get("weight_decay", 0.01)),
        "precision": values.get("precision", "bf16"), "device": values.get("device", "cuda"),
        "attention_backend": "sdpa",
        "conditioning_update": "floor-protect",
        "conditioning_update_state_policy": "main_adamw_moments_projected_floor_displacement_v1",
        "floor_focal_gamma": float(values.get("floor_focal_gamma", 2.0)),
        "feature_noise_std": float(values.get("rssi_noise_std", 0.03)),
        "feature_drop_rate": float(values.get("wap_drop_rate", 0.30)),
        "augmentation_space": "normalized_positive_features",
        "model": {"embed_dim": int(values.get("embed_dim", 160)), "depth": int(values.get("depth", 2)),
                  "num_heads": int(values.get("num_heads", 8)), "dropout": float(values.get("dropout", 0.1)),
                  "num_bins": int(values.get("num_bins", 100)), "mlp_ratio": int(values.get("mlp_ratio", 1)),
                  "qkv_bias": False, "qk_norm": True, "proj_bias": True},
        "optimizer": "AdamW", "scheduler": "CosineAnnealingLR", "scheduler_eta_min": 0.0,
        "gradient_clip_model_norm": 1.0, "uw_initial_log_scale": -0.5,
        "selection": "min_validation_mean_2d_error_m", "selection_tie_break": "earliest_epoch",
        "active_tasks": list(metadata["active_tasks"]), "prepared_hash": metadata["prepared_hash"],
        "deterministic_algorithms": True,
        "weighting_backend": BACKEND,
        "libmtl_upstream_commit": UPSTREAM_COMMIT,
    }
    if metadata["transform"]["norm_method"] != "mean_center":
        raise ValueError("Prepared data must use mean-centred RSSI normalization")
    if not 0 <= config["seed"] < 2**32:
        raise ValueError("Invalid seed")
    if min(config["epochs"], config["batch_size"]) < 1 or config["num_workers"] < 0:
        raise ValueError("Epochs and batch size must be positive; workers nonnegative")
    for name in ("lr", "weight_decay", "floor_focal_gamma", "feature_noise_std", "feature_drop_rate"):
        if not math.isfinite(config[name]) or config[name] < 0:
            raise ValueError(f"Invalid {name}")
    if config["lr"] <= 0 or config["feature_drop_rate"] > 1:
        raise ValueError("Learning rate must be positive; dropout probability must not exceed one")
    m = config["model"]
    if m["embed_dim"] < 2 or m["embed_dim"] % 2 or m["num_heads"] < 1 or m["embed_dim"] % m["num_heads"]:
        raise ValueError("Embedding dimension must be even and divisible by heads")
    if m["mlp_ratio"] < 1:
        raise ValueError("FFN ratio must be a positive integer")
    if m["depth"] < 1 or m["num_bins"] < 2 or not 0 <= m["dropout"] < 1:
        raise ValueError("Invalid model depth, bins or dropout")
    return config


def validate_device(device, precision):
    device = torch.device(device)
    if device.type not in ("cpu", "cuda"):
        raise ValueError("Only CPU and CUDA execution are supported")
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable; use --device cpu for a CPU check")
        # set_device requires an explicit index, even when .to('cuda') does not.
        if device.index is None:
            device = torch.device("cuda", torch.cuda.current_device())
        torch.cuda.set_device(device)
    if precision not in ("fp32", "bf16"):
        raise ValueError("Precision must be fp32 or bf16")
    if precision == "bf16" and (device.type != "cuda" or not torch.cuda.is_bf16_supported()):
        raise RuntimeError("BF16 requires a CUDA device with BF16 support; no silent precision fallback")
    return device


def build_model(metadata, config):
    m = config["model"]
    attn = {"num_heads": m["num_heads"], "qkv_bias": m["qkv_bias"], "qk_norm": m["qk_norm"],
            "porj_bias": m["proj_bias"], "attn_drop": m["dropout"], "proj_drop": m["dropout"]}
    encoder = {"input_dim": metadata["num_features"], "embed_dim": m["embed_dim"], "depth": m["depth"],
               "block_cfg": {"mlp_ratio": m["mlp_ratio"], "attn_cfg": attn},
               "dropout": m["dropout"]}
    head = {"dim": m["embed_dim"], "num_buildings": metadata["num_buildings"],
            "num_floors": metadata["num_floors"], "dropout": m["dropout"],
            "num_bins": m["num_bins"]}
    model = Witra(encoder, head)
    return model


class PreparedDataset(Dataset):
    def __init__(self, path, training=False, noise_std=0.03, drop_rate=0.30, seed=42):
        with np.load(path, allow_pickle=False) as archive:
            self.data = {key: archive[key].copy() for key in ("features", "coords", "coords_raw", "building", "floor", "row_id")}
        n = len(self.data["row_id"])
        if n == 0 or any(len(array) != n for array in self.data.values()):
            raise ValueError("Empty partition or inconsistent array lengths")
        if self.data["features"].ndim != 2 or self.data["coords"].shape != (n, 2) or self.data["coords_raw"].shape != (n, 2):
            raise ValueError("Invalid prepared array dimensions")
        if self.data["building"].shape != (n,) or self.data["floor"].shape != (n,):
            raise ValueError("Classification labels must be one dimensional")
        if self.data["row_id"].dtype.kind not in "US" or len(np.unique(self.data["row_id"])) != n:
            raise ValueError("Row IDs must be unique strings")
        for key, array in self.data.items():
            if key != "row_id" and not np.isfinite(array).all():
                raise ValueError(f"Non-finite prepared {key}")
        if any(self.data[key].dtype.kind not in "iu" for key in ("building", "floor")):
            raise ValueError("Class labels must be integers")
        self.training, self.noise_std, self.drop_rate = training, noise_std, drop_rate
        self.generator = torch.Generator().manual_seed(seed)

    def __len__(self):
        return len(self.data["row_id"])

    def __getitem__(self, index):
        features = torch.from_numpy(self.data["features"][index].copy()).float()
        if self.training:
            # This intentionally preserves the legacy FEATURE-space regularizer.
            # Positive normalized features are not a physical raw heard-AP mask.
            active = features > 0
            if self.noise_std:
                noise = torch.randn(features.shape, generator=self.generator) * self.noise_std
                features = (features + noise * active).clamp(0, 1)
            if self.drop_rate:
                drop = (torch.rand(features.shape, generator=self.generator) < self.drop_rate) & active
                features[drop] = 0
        return {"features": features, "coords": torch.as_tensor(self.data["coords"][index], dtype=torch.float32),
                "coords_raw": torch.as_tensor(self.data["coords_raw"][index], dtype=torch.float64),
                "building": int(self.data["building"][index]), "floor": int(self.data["floor"][index]),
                "row_id": str(self.data["row_id"][index])}


def _seed_worker(worker_id):
    info = get_worker_info()
    seed = info.seed % 2**32
    random.seed(seed)
    np.random.seed(seed)
    info.dataset.generator.manual_seed(info.seed)


def make_loader(npz_path, batch_size, seed, num_workers=0, training=False, noise_std=0.03, drop_rate=0.30):
    dataset = PreparedDataset(npz_path, training, noise_std, drop_rate, seed)
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(dataset, batch_size=batch_size, shuffle=training, num_workers=num_workers,
                      generator=generator, worker_init_fn=_seed_worker, drop_last=False,
                      persistent_workers=False, pin_memory=torch.cuda.is_available())


def _check_partition(loader, metadata):
    data = loader.dataset.data
    if data["features"].shape[1] != metadata["num_features"]:
        raise ValueError("Prepared AP count does not match metadata")
    for key, limit in (("building", metadata["num_buildings"]), ("floor", metadata["num_floors"])):
        if np.min(data[key]) < 0 or np.max(data[key]) >= limit:
            raise ValueError(f"{key} label outside training-defined class mapping")


def task_losses(outputs, batch, gamma=2.0):
    coord, floor, building = outputs[:3]
    ce = F.cross_entropy(floor.float(), batch["floor"], reduction="none")
    floor_loss = ((1 - torch.exp(-ce)).pow(gamma) * ce).mean()
    return {"coord": F.mse_loss(coord.float(), batch["coords"].float()),
            "building": F.cross_entropy(building.float(), batch["building"]), "floor": floor_loss}


def task_objective(losses, active_tasks, log_scales):
    # Reporting/reference scalar. UW training calls actual LibMTL.UW.backward.
    values = torch.stack([losses[name] for name in active_tasks])
    if log_scales.shape != values.shape:
        raise ValueError("UW scale count must equal the number of active tasks")
    return (0.5 * torch.exp(-log_scales) * values + 0.5 * log_scales).sum()


def _finite(tensor, name):
    if not torch.isfinite(tensor).all().item():
        raise FloatingPointError(f"Non-finite {name}; run invalid, no rows will be omitted")


def _finite_many(named_tensors):
    """One host synchronization on the successful path; retain precise failures."""
    values = list(named_tensors)
    if values and not torch.stack([torch.isfinite(value).all() for _, value in values]).all().item():
        for name, value in values:
            _finite(value, name)


def _autocast(device, precision):
    return torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=precision == "bf16")


@torch.no_grad()
def predict(model, loader, metadata, device, precision="fp32"):
    device = torch.device(device)
    _check_partition(loader, metadata)
    model.eval()
    _finite_many((f"model parameter {name}", parameter) for name, parameter in model.named_parameters())
    collected = {key: [] for key in ("row_id", "pred_xy", "true_xy", "floor_pred", "floor_true", "building_pred", "building_true", "floor_prob", "building_prob")}
    transform = metadata["transform"]
    origin = np.array([transform["min_x"], transform["min_y"]], dtype=np.float64)
    scale = float(transform["max_range"])
    if not np.isfinite(origin).all() or not math.isfinite(scale) or scale <= 0:
        raise ValueError("Invalid stored coordinate transform")
    for batch in loader:
        with _autocast(device, precision):
            coord, floor, building = model(batch["features"].to(device))
        _finite_many((("coordinate predictions", coord), ("floor logits", floor), ("building logits", building)))
        pred = coord.float().cpu().numpy().astype(np.float64) * scale + origin
        collected["row_id"].extend(batch["row_id"])
        for key, value in (("pred_xy", pred), ("true_xy", batch["coords_raw"].numpy()),
                           ("floor_pred", floor.argmax(-1).cpu().numpy()), ("floor_true", batch["floor"].numpy()),
                           ("building_pred", building.argmax(-1).cpu().numpy()), ("building_true", batch["building"].numpy()),
                           ("floor_prob", floor.float().softmax(-1).cpu().numpy()),
                           ("building_prob", building.float().softmax(-1).cpu().numpy())):
            collected[key].append(value)
    arrays = {key: np.asarray(value) if key == "row_id" else np.concatenate(value) for key, value in collected.items()}
    expected_ids = loader.dataset.data["row_id"].astype(str)
    if not np.array_equal(arrays["row_id"], expected_ids):
        raise ValueError("Evaluation omitted, duplicated or reordered IDs; use a nonshuffled evaluation loader")
    if any(not np.isfinite(value).all() for key, value in arrays.items() if key != "row_id"):
        raise FloatingPointError("Non-finite prediction output")
    errors = np.linalg.norm(arrays["pred_xy"] - arrays["true_xy"], axis=1)
    arrays["error_m"] = errors
    metrics = {"n_samples": int(len(errors)), "mean_error_m": float(errors.mean()),
               "median_error_m": float(np.median(errors)), "p90_error_m": float(np.quantile(errors, 0.9)),
               "building_accuracy_pct": float(100 * np.mean(arrays["building_pred"] == arrays["building_true"])),
               "floor_accuracy_pct": float(100 * np.mean(arrays["floor_pred"] == arrays["floor_true"]))}
    return metrics, arrays


def run_path(output_root, dataset, variant, seed):
    return Path(output_root) / dataset / variant / f"seed_{seed}"


def _software(device):
    return {"python": sys.version, "platform": platform.platform(), "torch": str(torch.__version__),
            "numpy": str(np.__version__), "cuda_runtime": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(), "device": str(device),
            "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else platform.processor() or "CPU",
            "libmtl_upstream_commit": UPSTREAM_COMMIT, "libmtl_uw_source": str(UW_SOURCE)}


def save_best_checkpoint(directory, checkpoint, is_best):
    """Keep only the validation-selected checkpoint as best.pt."""
    if not is_best:
        return None
    directory = Path(directory).resolve()
    temporary = directory / "best.pt.tmp"
    torch.save(checkpoint, temporary)
    os.replace(temporary, directory / "best.pt")
    return "best.pt"


def train(args):
    prepared = Path(args.prepared).resolve()
    metadata = load_metadata(prepared)
    config = resolve_config(args, metadata)
    device = validate_device(config["device"], config["precision"])
    for partition in ("train.npz", "val.npz"):
        verify_artifact(prepared, metadata, partition)
    directory = run_path(args.output_root, metadata["dataset"], config["variant"], config["seed"]).resolve()
    directory.mkdir(parents=True, exist_ok=False)
    try:
        relative_prepared = os.path.relpath(prepared, directory)
    except ValueError:
        relative_prepared = None
    run_id = f"{PROTOCOL}/{metadata['dataset']}/{config['variant']}/seed_{config['seed']}"
    manifest = {"protocol": PROTOCOL, "run_id": run_id, "status": "training", "created_utc": utc_now(),
                "config": config, "config_hash": digest_json(config), "prepared_hash": metadata["prepared_hash"],
                "prepared_relative_path": relative_prepared, "prepared_absolute_path": str(prepared),
                "code_hashes": source_hashes(), "software": _software(device)}
    write_json(directory / "manifest.json", manifest)
    started = time.perf_counter()
    try:
        seed_everything(config["seed"])
        model = build_model(metadata, config).to(device)
        manifest["model_parameters"] = sum(parameter.numel() for parameter in model.parameters())
        write_json(directory / "manifest.json", manifest)
        train_loader = make_loader(prepared / "train.npz", config["batch_size"], config["seed"], config["num_workers"],
                                   True, config["feature_noise_std"], config["feature_drop_rate"])
        val_loader = make_loader(prepared / "val.npz", config["batch_size"], config["seed"], config["num_workers"])
        _check_partition(train_loader, metadata)
        _check_partition(val_loader, metadata)
        if set(train_loader.dataset.data["row_id"]) & set(val_loader.dataset.data["row_id"]):
            raise ValueError("Training and validation row IDs overlap")
        weighting = create_weighting(config, device)
        log_scales = weighting.loss_scale
        print(f"MTL_BACKEND={config['weighting_backend']} ACTIVE_TASKS={config['active_tasks']} SOURCE={UW_SOURCE}", flush=True)
        # UW log-scales form a separate group and are not gradient-clipped.
        groups = [{"params": model.parameters()}, {"params": [log_scales]}]
        optimizer = torch.optim.AdamW(groups, lr=config["lr"], weight_decay=config["weight_decay"])
        protection = FloorUpdateProtection(model.multihead.floor_head, optimizer, config['gradient_clip_model_norm'])
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["epochs"], eta_min=0.0)
        history, best_error, selected_epoch = [], math.inf, None
        for epoch in range(1, config["epochs"] + 1):
            epoch_start = time.perf_counter()
            model.train()
            loss_names = ("coord", "floor", "building", "objective")
            # Match Python float accumulation in float64, transferring logs once per epoch.
            totals = torch.zeros(len(loss_names), dtype=torch.float64, device=device)
            n_samples = 0
            uw_backward_calls = 0
            protection_totals = torch.zeros(6, dtype=torch.float64, device=device)
            lr = float(optimizer.param_groups[0]["lr"])
            for batch in train_loader:
                batch = {key: value.to(device, non_blocking=True) if key in ("features", "coords", "building", "floor") else value
                         for key, value in batch.items()}
                optimizer.zero_grad(set_to_none=True)
                with _autocast(device, config["precision"]):
                    outputs = model(batch["features"])
                    losses = task_losses(outputs, batch, config["floor_focal_gamma"])
                    objective = task_objective(losses, config["active_tasks"], log_scales)
                    total = objective
                logged_losses = torch.stack([losses[name].detach() for name in loss_names[:-1]] + [total.detach()])
                _finite(logged_losses, "training losses/objective")
                # Algorithm 1: floor-loss gradient, UW backward, reference step,
                # clipping, main AdamW step, then the floor-head correction.
                floor_gradient = protection.floor_gradient(losses['floor'])
                floor_coefficient = 0.5 * torch.exp(-log_scales.detach()[config['active_tasks'].index('floor')])
                weighting.backward(torch.stack([losses[name].float() for name in config["active_tasks"]]))
                uw_backward_calls += 1
                protection.prepare_reference(floor_gradient, floor_coefficient)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config["gradient_clip_model_norm"], error_if_nonfinite=True)
                _finite(log_scales.grad, "UW gradient")
                optimizer.step()
                protection_totals += protection.apply()
                _finite(log_scales, "UW scales")
                count = len(batch["features"])
                totals += logged_losses.double() * count
                n_samples += count
            if n_samples != len(train_loader.dataset):
                raise RuntimeError("Training loader did not visit every prepared record")
            val_metrics, _ = predict(model, val_loader, metadata, device, config["precision"])
            scheduler.step()
            mean_losses = {name: value / n_samples for name, value in zip(loss_names, totals.cpu().tolist())}
            row = {"epoch": epoch, "lr": lr, "train_losses": mean_losses,
                   "validation": val_metrics, "elapsed_seconds": time.perf_counter() - epoch_start,
                   "uw_log_scales": log_scales.detach().cpu().tolist(),
                   "weighting_backend": config["weighting_backend"], "uw_backward_calls": uw_backward_calls}
            probes, adjusted, skipped, gap, margin, norm = protection_totals.cpu().tolist()
            row['floor_protection'] = {'steps': int(probes), 'adjusted_steps': int(adjusted),
                'skipped_tiny_floor_gradient': int(skipped), 'mean_positive_reference_gap': gap/probes,
                'mean_realized_reference_margin': margin/probes, 'mean_correction_norm': norm/probes,
                'state_policy': config['conditioning_update_state_policy']}
            if val_metrics["mean_error_m"] < best_error:
                best_error, selected_epoch = val_metrics["mean_error_m"], epoch
            checkpoint = {"protocol": PROTOCOL, "run_id": run_id, "config_hash": manifest["config_hash"],
                              "code_hashes": manifest["code_hashes"], "prepared_hash": metadata["prepared_hash"],
                              "config": config, "metadata": metadata, "epoch": epoch, "selected_epoch": epoch,
                              "best_epoch_so_far": selected_epoch,
                              "selection": config["selection"], "validation_metrics": val_metrics,
                              "model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(),
                              "scheduler_state_dict": scheduler.state_dict(),
                              "weighting_state_dict": {name: value.detach().cpu() for name, value in weighting.state_dict().items()},
                              "uw_log_scales": log_scales.detach().cpu()}
            row["checkpoint_file"] = save_best_checkpoint(directory, checkpoint, selected_epoch == epoch)
            row["elapsed_seconds"] = time.perf_counter() - epoch_start
            row["is_selected_checkpoint"] = selected_epoch == epoch
            history.append(row)
            write_json(directory / "history.json", history)
            print(json.dumps({"epoch": epoch, "validation": val_metrics, "selected_epoch": selected_epoch,
                              "epoch_seconds": row["elapsed_seconds"]}, allow_nan=False), flush=True)
        if selected_epoch is None or not (directory / "best.pt").is_file():
            raise RuntimeError("No valid checkpoint was selected")
        manifest.update({"status": "completed", "completed_utc": utc_now(), "elapsed_seconds": time.perf_counter() - started,
                         "completed_epochs": len(history),
                         "selected_epoch": selected_epoch, "selected_validation_metrics": history[selected_epoch - 1]["validation"],
                         "best_checkpoint_sha256": sha256_file(directory / "best.pt"), "history_sha256": sha256_file(directory / "history.json")})
        write_json(directory / "manifest.json", manifest)
        print(f"TRAIN_COMPLETE={directory}", flush=True)
        return directory
    except BaseException as error:
        manifest.update({"status": "failed", "failed_utc": utc_now(), "elapsed_seconds": time.perf_counter() - started,
                         "error_type": type(error).__name__, "error": str(error)})
        write_json(directory / "manifest.json", manifest)
        (directory / "failure.txt").write_text(traceback.format_exc(), encoding="utf-8")
        raise


def _load_completed_run(directory):
    directory = Path(directory).resolve()
    manifest = read_json(directory / "manifest.json")
    if manifest.get("status") != "completed" or manifest.get("protocol") != PROTOCOL:
        raise ValueError("Run is not a completed run of this protocol")
    if manifest.get("config_hash") != digest_json(manifest["config"]):
        raise ValueError("Run configuration hash mismatch")
    if manifest.get("code_hashes") != source_hashes():
        raise ValueError("Current source differs from the training source; use the exact saved code revision")
    if sha256_file(directory / "best.pt") != manifest.get("best_checkpoint_sha256"):
        raise ValueError("Missing, stale or modified selected checkpoint")
    if sha256_file(directory / "history.json") != manifest.get("history_sha256"):
        raise ValueError("Training history hash mismatch")
    return directory, manifest


def _prepared_for_run(directory, manifest, override=None):
    if override:
        prepared = Path(override).resolve()
    elif manifest.get("prepared_relative_path"):
        prepared = (directory / manifest["prepared_relative_path"]).resolve()
    else:
        prepared = Path(manifest["prepared_absolute_path"])
    metadata = load_metadata(prepared)
    if metadata["prepared_hash"] != manifest["prepared_hash"] or metadata["prepared_hash"] != manifest["config"]["prepared_hash"]:
        raise ValueError("Evaluation preprocessing/split does not match the training manifest")
    return prepared, metadata


def evaluate(args):
    directory, manifest = _load_completed_run(args.run_dir)
    if (directory / "metrics.json").exists():
        raise FileExistsError("This run already has final test metrics; existing evaluation is preserved")
    prepared, metadata = _prepared_for_run(directory, manifest, getattr(args, "prepared", None))
    config = manifest["config"]
    device = validate_device(args.device or config["device"], config["precision"])
    # weights_only avoids arbitrary pickle execution. Only this runner's tensor/
    # primitive checkpoints are supported; there is no permissive load fallback.
    checkpoint = torch.load(directory / "best.pt", map_location="cpu", weights_only=True)
    for key in ("protocol", "run_id", "config_hash", "code_hashes", "prepared_hash", "selected_epoch"):
        if checkpoint.get(key) != manifest.get(key):
            raise ValueError(f"Checkpoint belongs to a different run: {key}")
    if checkpoint.get("config") != config or checkpoint.get("metadata") != metadata:
        raise ValueError("Checkpoint configuration/preprocessing state mismatch")
    if checkpoint.get("validation_metrics") != manifest.get("selected_validation_metrics"):
        raise ValueError("Selected-checkpoint validation metrics disagree with manifest")
    validate_checkpoint_weighting(checkpoint, config)
    seed_everything(config["seed"])
    model = build_model(metadata, config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    # This is the first point at which test.npz is opened by the runner.
    verify_artifact(prepared, metadata, "test.npz")
    loader = make_loader(prepared / "test.npz", config["batch_size"], config["seed"], config["num_workers"])
    metrics, arrays = predict(model, loader, metadata, device, config["precision"])
    predictions = directory / "predictions.csv"
    temporary = directory / "predictions.csv.tmp"
    floor_classes = metadata["transform"]["floor_classes"]
    building_classes = metadata["transform"]["building_classes"]
    columns = ["row_id", "true_x_m", "true_y_m", "pred_x_m", "pred_y_m", "error_m",
               "true_floor_index", "pred_floor_index", "true_floor_label", "pred_floor_label",
               "true_building_index", "pred_building_index", "true_building_label", "pred_building_label"]
    columns += [f"floor_probability_index_{i}" for i in range(metadata["num_floors"])]
    columns += [f"building_probability_index_{i}" for i in range(metadata["num_buildings"])]
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(columns)
        for i, row_id in enumerate(arrays["row_id"]):
            ft, fp = int(arrays["floor_true"][i]), int(arrays["floor_pred"][i])
            bt, bp = int(arrays["building_true"][i]), int(arrays["building_pred"][i])
            writer.writerow([row_id, *arrays["true_xy"][i], *arrays["pred_xy"][i], arrays["error_m"][i],
                             ft, fp, floor_classes[ft], floor_classes[fp], bt, bp, building_classes[bt], building_classes[bp],
                             *arrays["floor_prob"][i], *arrays["building_prob"][i]])
    os.replace(temporary, predictions)
    result = {"protocol": PROTOCOL, "run_id": manifest["run_id"], "dataset": config["dataset"],
              "variant": config["variant"], "seed": config["seed"], "config_hash": manifest["config_hash"],
              "prepared_hash": metadata["prepared_hash"], "checkpoint_sha256": manifest["best_checkpoint_sha256"],
              "selected_epoch": manifest["selected_epoch"], "selected_validation_metrics": manifest["selected_validation_metrics"],
              "test": metrics, "predictions_sha256": sha256_file(predictions), "evaluated_utc": utc_now(),
              "evaluation_software": _software(device), "class_mapping": {"floor": floor_classes, "building": building_classes}}
    write_json(directory / "metrics.json", result)
    print("EVALUATION_COMPLETE=" + json.dumps(result, ensure_ascii=False, allow_nan=False), flush=True)
    return result


def _mean_sd(values):
    array = np.asarray(values, dtype=np.float64)
    return {"mean": float(array.mean()), "sample_sd": float(array.std(ddof=1)) if len(array) > 1 else None, "n_seeds": len(array)}


def aggregate(args):
    """Mean and sample SD of MAIN test metrics across training seeds."""
    if len(set(args.seeds)) != len(args.seeds) or len(set(args.datasets)) != len(args.datasets):
        raise ValueError("Expected dataset/seed lists must not contain duplicates")
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError("Aggregate output exists; choose a new output file")
    records, summaries = [], []
    for dataset in args.datasets:
        reference_config, reference_prepared, grid = None, None, {}
        for seed in args.seeds:
            directory, manifest = _load_completed_run(run_path(args.output_root, dataset, VARIANT, seed))
            config = manifest["config"]
            if (config["dataset"], config["variant"], config["seed"]) != (dataset, VARIANT, seed):
                raise ValueError("Run path and manifest identity disagree")
            # Every setting except the RNG seed must match across seeds.
            shared = {key: value for key, value in config.items() if key not in ("seed", "prepared_hash")}
            _, prepared_metadata = _prepared_for_run(directory, manifest)
            comparison_data = {key: prepared_metadata[key] for key in
                               ("dataset", "source_hashes", "splits", "counts", "ancestry", "transform", "config", "code_hashes")}
            data_signature = digest_json(comparison_data)
            if reference_config is None:
                reference_config, reference_prepared = shared, data_signature
            elif shared != reference_config or data_signature != reference_prepared:
                raise ValueError("Runs do not share the same dataset/split/training protocol")
            result = read_json(directory / "metrics.json")
            if (result.get("protocol"), result.get("dataset"), result.get("variant"), result.get("seed")) != (PROTOCOL, dataset, VARIANT, seed):
                raise ValueError("Evaluation identity/protocol mismatch")
            if result["test"].get("n_samples") != prepared_metadata["counts"]["test"]:
                raise ValueError("Evaluation test row count differs from prepared data")
            for result_key, manifest_key in (("run_id", "run_id"), ("config_hash", "config_hash"),
                                              ("prepared_hash", "prepared_hash"), ("checkpoint_sha256", "best_checkpoint_sha256"),
                                              ("selected_epoch", "selected_epoch")):
                if result.get(result_key) != manifest.get(manifest_key):
                    raise ValueError(f"Evaluation/manifest mismatch: {result_key}")
            if result["predictions_sha256"] != sha256_file(directory / "predictions.csv"):
                raise ValueError("Saved predictions have changed")
            if any(not math.isfinite(result["test"][key]) for key in METRIC_NAMES):
                raise ValueError("Non-finite metric in aggregate input")
            records.append(result)
            grid[seed] = result
        summaries.append({"dataset": dataset, "metrics": {
            name: _mean_sd([grid[seed]["test"][name] for seed in args.seeds]) for name in METRIC_NAMES}})
    result = {"protocol": PROTOCOL, "created_utc": utc_now(),
              "expected_grid": {"datasets": args.datasets, "seeds": args.seeds},
              "n_expected": len(args.datasets) * len(args.seeds), "n_completed": len(records),
              "interpretation": "Mean and sample SD across per-seed metrics; no ensemble and no sample-times-seed independence assumption.",
              "runs": records, "summary": summaries}
    output.parent.mkdir(parents=True, exist_ok=True)
    write_json(output, result)
    print(f"AGGREGATE_COMPLETE={output}", flush=True)
    return result


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    t = sub.add_parser("train", help="Train using prepared train and validation partitions only")
    t.add_argument("--prepared", required=True)
    t.add_argument("--output-root", required=True)
    t.add_argument("--seed", type=int, default=42)
    t.add_argument("--epochs", type=int, default=300)
    t.add_argument("--batch-size", type=int, help="Default: UJI/UTS 32, TUT 16")
    t.add_argument("--num-workers", type=int, default=0)
    t.add_argument("--device", default="cuda")
    t.add_argument("--precision", choices=("fp32", "bf16"), default="bf16")
    t.add_argument("--lr", type=float, help="Default: UJI 8e-4, UTS 1e-4, TUT 1e-4")
    t.add_argument("--weight-decay", type=float, default=0.01)
    t.add_argument("--floor-focal-gamma", type=float, default=2.0)
    t.add_argument("--rssi-noise-std", type=float, default=0.03)
    t.add_argument("--wap-drop-rate", type=float, default=0.30)
    t.add_argument("--embed-dim", type=int, default=160)
    t.add_argument("--depth", type=int, default=2)
    t.add_argument("--mlp-ratio", type=int, default=1, help="FFN hidden width / embedding width")
    t.add_argument("--num-heads", type=int, default=8)
    t.add_argument("--dropout", type=float, default=0.1)
    t.add_argument("--num-bins", type=int, default=100)
    e = sub.add_parser("evaluate", help="Evaluate the selected checkpoint once on the supplied test partition")
    e.add_argument("--run-dir", required=True)
    e.add_argument("--prepared", help="Optional relocated prepared directory; its digest must match the run")
    e.add_argument("--device", help="Execution device override; training precision is preserved")
    a = sub.add_parser("aggregate", help="Summarize the test metrics of completed seeds")
    a.add_argument("--output-root", required=True)
    a.add_argument("--datasets", nargs="+", choices=("uji", "uts", "tampere"), required=True)
    a.add_argument("--seeds", nargs="+", type=int, required=True)
    a.add_argument("--output", required=True)
    return p


def main():
    args = parser().parse_args()
    return {"train": train, "evaluate": evaluate, "aggregate": aggregate}[args.command](args)


if __name__ == "__main__":
    main()
