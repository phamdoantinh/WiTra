"""
MTL Adapter: bridges LibMTL's AbsWeighting into Witra model.

Design (mirrors LibMTL Trainer._prepare_model):
    class MTLAdapter(UW):
        get_share_params() → model.encoder.parameters()
"""
from pathlib import Path
import sys
import numpy as np
import torch
import torch.nn as nn

# ── LibMTL path ──────────────────────────────────────────────────────────────
_LIBMTL_ROOT = Path(__file__).parent.parent / "LibMTL"
if not _LIBMTL_ROOT.exists():
    raise FileNotFoundError(
        f"LibMTL not found at {_LIBMTL_ROOT}. "
        "Clone: git clone https://github.com/median-research-group/LibMTL.git"
    )
if str(_LIBMTL_ROOT) not in sys.path:
    sys.path.insert(0, str(_LIBMTL_ROOT))

from LibMTL.weighting.UW import UW

# ── Constants required by mtl_trainer ─────────────────────────────────────────
SUPPORTED_STRATEGIES = ("UW",)
_HAS_LEARNABLE_PARAMS = {"UW"}
_NEEDS_LOSS_BUFFER = set()
_NEEDS_UPDATE_W = set()
_COMPILE_INCOMPATIBLE = set()


def create_mtl_adapter(model: nn.Module, device: torch.device,
                       strategy: str, num_epochs: int) -> UW:
    """
    Factory: dynamically create an MTLAdapter class inheriting from UW.
    """
    if strategy != "UW":
        raise ValueError(f"Only 'UW' strategy is supported now, got '{strategy}'")

    if not hasattr(model, "encoder"):
        raise AttributeError(
            "model has no 'encoder' attribute. "
            "MTLAdapter expects Witra with model.encoder as shared backbone."
        )

    # ── Dynamic class ─────────────────────────────────────────────────────────
    class MTLAdapter(UW):
        """
        MTLAdapter wraps Witra and satisfies the AbsWeighting interface for UW.
        """
        TASK_NAMES = ["coord", "building", "floor"]

        def __init__(self):
            super().__init__()
            self.wrapped_model = model
            self.device = device
            self.task_num = 3
            self.task_name = self.TASK_NAMES
            self.rep_grad = False                  # weighting on shared params
            self.epoch = 0
            self.train_loss_buffer = np.zeros([self.task_num, num_epochs])
            
            # Learnable params (UW: log-variance)
            self.init_param()

        # ── AbsWeighting required interface ──────────────────────────────────

        def get_share_params(self):
            """Shared backbone = Witra encoder only (not task heads)."""
            return self.wrapped_model.encoder.parameters()

        def zero_grad_share_params(self):
            for p in self.get_share_params():
                if p.grad is not None:
                    p.grad.zero_()

    adapter = MTLAdapter()
    adapter.to(device)
    return adapter
