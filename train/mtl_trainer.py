"""
MTLTrainer: drop-in replacement for Trainer with LibMTL adaptive loss weighting.
"""
import os
import time
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from typing import Dict, Optional

try:
    from train.train_core import Trainer
    from train.mtl_adapter import (
        create_mtl_adapter,
        SUPPORTED_STRATEGIES,
        _HAS_LEARNABLE_PARAMS,
    )
    from train.mtl_losses import CoordLoss, BuildingLoss, FloorLoss
except (ImportError, ModuleNotFoundError):
    from train_core import Trainer
    from mtl_adapter import (
        create_mtl_adapter,
        SUPPORTED_STRATEGIES,
        _HAS_LEARNABLE_PARAMS,
    )
    from mtl_losses import CoordLoss, BuildingLoss, FloorLoss


class MTLTrainer(Trainer):
    """
    LibMTL-powered trainer for WiFi indoor positioning.

    Replaces the fixed weighted-sum loss in train_core.Trainer with one
    of the LibMTL weighting strategies (defaulting to UW).
    """

    def __init__(
        self,
        model: nn.Module,
        device: str = "cuda",
        logger = None,
        norm_params: Optional[float] = None,
        mtl_strategy: str = "UW",
        num_epochs: int = 100,
        mtl_kwargs: Optional[Dict] = None,
        floor_focal_gamma: float = 2.0,
        floor_acc_weight: float = 5.0,
        use_amp: bool = True,
        use_compile: bool = True,
        w_diff: float = 0.01,
        w_sim: float = 0.0,
        w_sp_diff: float = 0.01,
    ):
        # ── Validate BEFORE anything else — no silent fallback ────────────────
        if mtl_strategy not in SUPPORTED_STRATEGIES:
            raise ValueError(
                f"mtl_strategy='{mtl_strategy}' is not supported.\n"
                f"Choose from: {SUPPORTED_STRATEGIES}"
            )

        super().__init__(
            model       = model,
            device      = device,
            logger      = logger,
            norm_params = norm_params,
        )
        self.mtl_strategy = mtl_strategy
        self.mtl_kwargs   = mtl_kwargs or {}
        self.num_epochs   = num_epochs
        self.use_amp      = use_amp and torch.cuda.is_available()

        # ── AMP (bf16) — RTX 5060 Ti has native bf16, no GradScaler needed ───
        self._amp_dtype = torch.bfloat16  # bf16: no overflow, no scaler needed

        # ── torch.compile ────────────────────────────────────────────────────
        if use_compile and self.device.type == "cuda":
            try:
                # Disable donated buffers to allow retain_graph=True with torch.compile
                try:
                    import torch._functorch.config as functorch_config
                    functorch_config.donated_buffer = False
                except (ImportError, AttributeError):
                    pass
                self.model = torch.compile(self.model, mode="reduce-overhead")
                msg = "torch.compile enabled (mode=reduce-overhead, donated_buffer=False)"
                (self.logger.info if self.logger else print)(msg)
            except Exception as e:
                msg = f"torch.compile failed, falling back to eager: {e}"
                (self.logger.warning if self.logger else print)(msg)

        # ── Per-task loss functions (AbsLoss) ─────────────────────────────────
        self.coord_loss_fn    = CoordLoss()
        self.building_loss_fn = BuildingLoss()
        self.floor_loss_fn    = FloorLoss(
            gamma=floor_focal_gamma,
        )
        self.floor_acc_weight = floor_acc_weight  # for composite checkpoint metric
        self.w_diff           = w_diff
        self.w_sim            = w_sim
        self.w_sp_diff        = w_sp_diff

        msg = (
            f"  FloorLoss: Focal(gamma={floor_focal_gamma})"
            f" | floor_acc_weight={floor_acc_weight}"
        )
        (self.logger.info if self.logger else print)(msg)

        # ── MTL adapter ───────────────────────────────────────────────────────
        self.mtl_adapter = create_mtl_adapter(
            model      = self.model,
            device     = self.device,
            strategy   = mtl_strategy,
            num_epochs = num_epochs,
        )

        msg = (
            f"MTLTrainer initialized | strategy={mtl_strategy} "
            f"| kwargs={self.mtl_kwargs} | amp={self.use_amp}"
        )
        (self.logger.info if self.logger else print)(msg)

    # ── setup_training ────────────────────────────────────────────────────────
    def setup_training(
        self,
        learning_rate: float = 1e-4,
        weight_decay: float = 0.01,
        **_ignored,           # absorb coord/building/floor_loss_weight kwargs
    ):
        """
        Build optimizer over model params + mtl_adapter learnable params.

        coord/building/floor_loss_weight kwargs are silently ignored —
        weighting is handled by mtl_adapter.backward().
        """
        param_groups = [{"params": self.model.parameters(), "lr": learning_rate}]

        if self.mtl_strategy in _HAS_LEARNABLE_PARAMS:
            # MTLAdapter is nn.Module and stores wrapped_model as attribute,
            # so .parameters() returns model params PLUS adapter-specific params.
            # Must exclude model params to avoid "appears in more than one group".
            model_param_ids = {id(p) for p in self.model.parameters()}
            mtl_only_params = [
                p for p in self.mtl_adapter.parameters()
                if id(p) not in model_param_ids
            ]
            if mtl_only_params:
                param_groups.append({"params": mtl_only_params, "lr": learning_rate})
                msg = (
                    f"  MTL learnable params added to optimizer "
                    f"({sum(p.numel() for p in mtl_only_params)} params)"
                    f"  [e.g. UW loss_scale x{len(mtl_only_params)}]"
                )
                (self.logger.info if self.logger else print)(msg)

        self.optimizer = torch.optim.AdamW(
            param_groups,
            lr           = learning_rate,
            weight_decay = weight_decay,
        )

        # Store weights = 1.0 so parent evaluate() works unchanged
        self.coord_loss_weight    = 1.0
        self.building_loss_weight = 1.0
        self.floor_loss_weight    = 1.0

        msg = (
            f"Training setup | lr={learning_rate} wd={weight_decay} "
            f"strategy={self.mtl_strategy}"
        )
        (self.logger.info if self.logger else print)(msg)

    # ── train_epoch ───────────────────────────────────────────────────────────
    def train_epoch(self, train_loader, epoch: int) -> Dict[str, float]:
        """
        One training epoch with MTL backward.

        Replaces parent's fixed-weight loss computation with:
            losses_tensor = [coord_loss, building_loss, floor_loss]
            mtl_adapter.backward(losses_tensor, **mtl_kwargs)

        AbsWeighting internals:
            _compute_grad_dim()     pre-compute grad dimension
            _compute_grad()         backward each task, collect grad vectors
            _reset_grad()           inject re-weighted gradient
        """
        self.model.train()
        total_coord    = 0.0
        total_building = 0.0
        total_floor    = 0.0
        total_diff     = 0.0
        total_sim      = 0.0
        total_sp_diff  = 0.0
        num_batches    = 0

        pbar = tqdm(
            train_loader,
            desc   = f"Epoch {epoch} [{self.mtl_strategy}]",
            colour = "cyan",
            leave  = False,
            disable=None,
        )

        for batch in pbar:
            features = batch["rssi features"].to(self.device, non_blocking=True)
            coords   = batch["coords"].to(self.device, non_blocking=True)
            building = batch["building"].to(self.device, non_blocking=True)
            floor    = batch["floor"].to(self.device, non_blocking=True)

            # ── Forward under bf16 autocast (backward stays fp32) ────────────
            with torch.autocast(device_type="cuda", dtype=self._amp_dtype,
                                enabled=self.use_amp):
                coord_pred, floor_pred, building_pred, aux_losses = self.model(features, return_ortho_loss=True)
                # losses computed inside autocast — still fp32 scalars after .item()
                coord_loss    = self.coord_loss_fn._update_loss(coord_pred.float(), coords)
                building_loss = self.building_loss_fn._update_loss(building_pred.float(), building)
                floor_loss    = self.floor_loss_fn._update_loss(floor_pred.float(), floor)

            # Stack → [3] tensor fp32, still attached to graph
            losses_tensor = torch.stack([coord_loss, building_loss, floor_loss])

            self.optimizer.zero_grad()
            
            # ── Backpropagate Shared-Private Auxiliary Loss directly on shared backbone FIRST ──
            # We use retain_graph=True so that LibMTL can subsequently backward through the shared graph.
            if aux_losses is not None:
                l_diff, l_sim, l_sp_diff = aux_losses
                loss_aux = self.w_diff * l_diff + self.w_sim * l_sim + self.w_sp_diff * l_sp_diff
                if loss_aux > 0:
                    loss_aux.backward(retain_graph=True)

            # MTL backward: weighting class computes & injects re-weighted grads (fp32)
            self.mtl_adapter.backward(losses_tensor, **self.mtl_kwargs)

            # Gradient clipping on full model
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()



            # Accumulate scalar values
            c_val = coord_loss.item()
            b_val = building_loss.item()
            f_val = floor_loss.item()
            diff_val = l_diff.item() if aux_losses is not None else 0.0
            sim_val = l_sim.item() if aux_losses is not None else 0.0
            sp_diff_val = l_sp_diff.item() if aux_losses is not None else 0.0

            total_coord    += c_val
            total_building += b_val
            total_floor    += f_val
            total_diff     += diff_val
            total_sim      += sim_val
            total_sp_diff  += sp_diff_val
            num_batches    += 1

            pbar.set_postfix(
                {
                    "crd":   f"{c_val:.4f}",
                    "bld":   f"{b_val:.4f}",
                    "flr":   f"{f_val:.4f}",
                    "diff":  f"{diff_val:.4f}",
                    "sim":   f"{sim_val:.4f}",
                }
            )

        # Reset AbsLoss per-epoch accumulators
        for fn in (self.coord_loss_fn, self.building_loss_fn, self.floor_loss_fn):
            fn._reinit()

        return {
            "loss":          (total_coord + total_building + total_floor) / num_batches,
            "coord_loss":    total_coord    / num_batches,
            "building_loss": total_building / num_batches,
            "floor_loss":    total_floor    / num_batches,
            "diff_loss":     total_diff     / num_batches,
            "sim_loss":      total_sim      / num_batches,
            "sp_diff_loss":  total_sp_diff  / num_batches,
        }

    # ── fit ───────────────────────────────────────────────────────────────────
    def fit(
        self,
        train_loader,
        val_loader,
        test_loader=None,
        num_epochs: int = 100,
        save_dir: str   = "./checkpoints",
        save_every: int = 10,
        test_every: int = 10,
    ) -> Dict:
        """
        Training loop. Extends parent by:
          1. Setting mtl_adapter.epoch before each train epoch
          2. Updating mtl_adapter.train_loss_buffer after each train epoch
             (required by DWA and GradNorm)
        """
        best_val_error = float("inf")

        for epoch in range(1, num_epochs + 1):
            start_time = time.time()

            # ── Tell adapter current epoch (DWA, GradNorm need this) ──────────
            self.mtl_adapter.epoch = epoch - 1   # 0-indexed like LibMTL

            # ── Train ─────────────────────────────────────────────────────────
            train_metrics = self.train_epoch(train_loader, epoch)



            # ── Validate ──────────────────────────────────────────────────────
            val_metrics = self.evaluate(val_loader, split="val")

            # ── LR scheduler ─────────────────────────────────────────────────
            if self.scheduler:
                self.scheduler.step()

            epoch_time = time.time() - start_time

            # ── History ───────────────────────────────────────────────────────
            self.history["train_loss"].append(train_metrics["loss"])
            self.history["train_coord_loss"].append(train_metrics["coord_loss"])
            self.history["train_building_loss"].append(train_metrics["building_loss"])
            self.history["train_floor_loss"].append(train_metrics["floor_loss"])
            self.history["train_diff_loss"].append(train_metrics["diff_loss"])
            self.history["train_sim_loss"].append(train_metrics["sim_loss"])
            self.history["train_sp_diff_loss"].append(train_metrics["sp_diff_loss"])
            
            self.history["val_loss"].append(val_metrics["loss"])
            self.history["val_coord_loss"].append(val_metrics["coord_loss"])
            self.history["val_building_loss"].append(val_metrics["building_loss"])
            self.history["val_floor_loss"].append(val_metrics["floor_loss"])
            self.history["val_diff_loss"].append(val_metrics["diff_loss"])
            self.history["val_sim_loss"].append(val_metrics["sim_loss"])
            self.history["val_sp_diff_loss"].append(val_metrics["sp_diff_loss"])
            self.history["val_coord_error"].append(val_metrics["coord_error_mean"])
            self.history["val_building_acc"].append(val_metrics["building_acc"])
            self.history["val_floor_acc"].append(val_metrics["floor_acc"])

            # ── Log ───────────────────────────────────────────────────────────
            log_msg = (
                f"\nEpoch {epoch}/{num_epochs} ({epoch_time:.1f}s) [{self.mtl_strategy}]\n"
                f"  Train: loss={train_metrics['loss']:.4f} (coord={train_metrics['coord_loss']:.4f}, bld={train_metrics['building_loss']:.4f}, flr={train_metrics['floor_loss']:.4f}, diff={train_metrics['diff_loss']:.4f}, sim={train_metrics['sim_loss']:.4f}, sp={train_metrics['sp_diff_loss']:.4f})\n"
                f"  Valid: loss={val_metrics['loss']:.4f} (coord={val_metrics['coord_loss']:.4f}, bld={val_metrics['building_loss']:.4f}, flr={val_metrics['floor_loss']:.4f}, diff={val_metrics['diff_loss']:.4f}, sim={val_metrics['sim_loss']:.4f}, sp={val_metrics['sp_diff_loss']:.4f})\n"
                f"  Metrics: error={val_metrics['coord_error_mean']:.2f}m (med={val_metrics['coord_error_median']:.2f}m) | bld={val_metrics['building_acc']:.2f}% | flr={val_metrics['floor_acc']:.2f}%"
            )
            (self.logger.info if self.logger else print)(log_msg)

            # Log mtl learnable params if any
            if self.mtl_strategy in _HAS_LEARNABLE_PARAMS:
                if hasattr(self.mtl_adapter, "loss_scale"):
                    scale_vals = self.mtl_adapter.loss_scale.detach().cpu().numpy()
                    scale_msg = (
                        f"  MTL loss_scale: coord={scale_vals[0]:.4f}"
                        f" bld={scale_vals[1]:.4f} flr={scale_vals[2]:.4f}"
                    )
                    (self.logger.info if self.logger else print)(scale_msg)

            # ── Best model (floor-aware composite metric) ─────────────────────
            floor_acc_pct  = val_metrics["floor_acc"]           # e.g. 93.52
            composite_score = (
                val_metrics["coord_error_mean"]
                - self.floor_acc_weight * (floor_acc_pct / 100.0)
            )

            if composite_score < best_val_error:
                best_val_error = composite_score
                best_path = os.path.join(save_dir, "best_model.pth")
                torch.save(
                    {
                        "epoch":              epoch,
                        "model_state_dict":   self.model.state_dict(),
                        "optimizer_state_dict": self.optimizer.state_dict(),
                        "val_error":          val_metrics["coord_error_mean"],
                        "val_floor_acc":      floor_acc_pct,
                        "composite_score":    composite_score,
                        "max_range":          self.norm_params,
                        "mtl_strategy":       self.mtl_strategy,
                        "mtl_adapter_state":  self.mtl_adapter.state_dict(),
                    },
                    best_path,
                )
                msg = (
                    f"  ✓ Best model saved"
                    f" (error={val_metrics['coord_error_mean']:.2f}m"
                    f" | floor={floor_acc_pct:.2f}%"
                    f" | score={composite_score:.4f})"
                )
                (self.logger.info if self.logger else print)(msg)

            # ── Checkpoint ────────────────────────────────────────────────────
            if epoch % save_every == 0:
                ckpt_path = os.path.join(save_dir, f"checkpoint_epoch_{epoch}.pth")
                torch.save(
                    {
                        "epoch":              epoch,
                        "model_state_dict":   self.model.state_dict(),
                        "optimizer_state_dict": self.optimizer.state_dict(),
                        "val_error":          val_metrics["coord_error_mean"],
                        "max_range":          self.norm_params,
                        "mtl_strategy":       self.mtl_strategy,
                        "mtl_adapter_state":  self.mtl_adapter.state_dict(),
                    },
                    ckpt_path,
                )
                (self.logger.info if self.logger else print)(
                    f"  Checkpoint saved: epoch_{epoch}.pth"
                )

            # ── Test ──────────────────────────────────────────────────────────
            if test_loader is not None and epoch % test_every == 0:
                (self.logger.info if self.logger else print)(
                    f"\nTesting at epoch {epoch}..."
                )
                test_metrics = self.evaluate(test_loader, split="test")
                test_msg = (
                    f"  Error: {test_metrics['coord_error_mean']:.2f}m"
                    f" (med: {test_metrics['coord_error_median']:.2f}m)"
                    f" | Building: {test_metrics['building_acc']:.2f}%"
                    f" | Floor: {test_metrics['floor_acc']:.2f}%"
                )
                (self.logger.info if self.logger else print)(test_msg)

        msg = (
            f"\nTraining complete [{self.mtl_strategy}]!"
            f" Best composite score: {best_val_error:.4f}"
            f" (= coord_error - {self.floor_acc_weight} × floor_acc)"
        )
        (self.logger.info if self.logger else print)(msg)
        return self.history

    def load_checkpoint(self, path: str, load_optimizer: bool = True):
        """Override to load MTL adapter parameters from checkpoint."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        super().load_checkpoint(path, load_optimizer=load_optimizer)
        if "mtl_adapter_state" in checkpoint:
            self.mtl_adapter.load_state_dict(checkpoint["mtl_adapter_state"])
            msg = f"Loaded MTL adapter ({self.mtl_strategy}) parameters from checkpoint."
            (self.logger.info if self.logger else print)(msg)
        elif "log_vars" in checkpoint:
            # Cross-compatibility: load from native log_vars format into LibMTL UW loss_scale parameter
            self.mtl_adapter.loss_scale.data.copy_(torch.tensor(checkpoint["log_vars"], device=self.device))
            msg = f"Loaded UW log-variance parameters into MTL adapter: {checkpoint['log_vars']}"
            (self.logger.info if self.logger else print)(msg)
