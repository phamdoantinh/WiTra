"""
MTL Adapter: bridges LibMTL's AbsWeighting into Witra model.

Design (mirrors LibMTL Trainer._prepare_model):
    class MTLAdapter(weighting_cls):  ← dynamic factory, same pattern as LibMTL
        get_share_params() → model.encoder.parameters()

Supported strategies:
  EW, UW, GradNorm, DWA, PCGrad          ← original
  CAGrad, MGDA, FAMO, GradVac            ← added
No silent fallback: unsupported strategy → ValueError immediately.
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

from LibMTL.weighting.abstract_weighting import AbsWeighting

# ── Supported methods ─────────────────────────────────────────────────────────
SUPPORTED_STRATEGIES = (
    "EW", "UW", "GradNorm", "DWA", "PCGrad",   # original
    "CAGrad", "MGDA", "FAMO", "GradVac",        # added
)

# These weighting classes have learnable nn.Parameters via init_param()
_HAS_LEARNABLE_PARAMS = {"UW", "GradNorm", "FAMO"}

# These read train_loss_buffer across epochs
_NEEDS_LOSS_BUFFER = {"DWA", "GradNorm"}

# FAMO needs update_w() called AFTER optimizer.step() each batch
_NEEDS_UPDATE_W = {"FAMO"}

# Strategies using retain_graph=True → incompatible with torch.compile donated buffers
_COMPILE_INCOMPATIBLE = {"GradNorm", "PCGrad", "CAGrad", "MGDA", "GradVac"}


def _import_weighting_class(strategy: str):
    """Import and return the LibMTL weighting class for *strategy*.

    Raises ValueError immediately if strategy not supported.
    Does NOT fall back to EW.
    """
    if strategy not in SUPPORTED_STRATEGIES:
        raise ValueError(
            f"mtl_strategy='{strategy}' is not supported.\n"
            f"Choose from: {SUPPORTED_STRATEGIES}"
        )
    import importlib
    mod = importlib.import_module(f"LibMTL.weighting.{strategy}")
    cls = getattr(mod, strategy, None)
    if cls is None:
        raise ImportError(
            f"Class '{strategy}' not found in LibMTL.weighting.{strategy}. "
            "Check LibMTL installation."
        )
    return cls


def create_mtl_adapter(model: nn.Module, device: torch.device,
                       strategy: str, num_epochs: int) -> AbsWeighting:
    """
    Factory: dynamically create an MTLAdapter class inheriting from the
    specific LibMTL weighting class (same pattern as LibMTL's MTLmodel).

    Args:
        model      : Witra instance (must have .encoder attribute)
        device     : torch.device
        strategy   : one of SUPPORTED_STRATEGIES
        num_epochs : total training epochs (pre-allocates loss buffer)

    Returns:
        MTLAdapter instance (nn.Module subclass)
    """
    if not hasattr(model, "encoder"):
        raise AttributeError(
            "model has no 'encoder' attribute. "
            "MTLAdapter expects Witra with model.encoder as shared backbone."
        )

    weighting_cls = _import_weighting_class(strategy)

    # ── Dynamic class ─────────────────────────────────────────────────────────
    # Mirrors LibMTL: class MTLmodel(architecture_class, weighting_class)
    class MTLAdapter(weighting_cls):
        """
        MTLAdapter wraps Witra and satisfies the AbsWeighting interface:

        Required attributes (set in __init__):
            task_num, task_name, device, rep_grad,
            epoch, train_loss_buffer

        Required methods (from AbsWeighting, called by weighting subclass):
            get_share_params()       → encoder params
            zero_grad_share_params() → zero encoder grads

        backward(**kwargs) is inherited from weighting_cls
            (GradNorm, UW, PCGrad, DWA, EW …)
        """

        TASK_NAMES = ["coord", "building", "floor"]

        def __init__(self):
            super().__init__()                     # AbsWeighting → nn.Module
            self.wrapped_model = model
            self.device = device
            self.task_num = 3
            self.task_name = self.TASK_NAMES
            self.rep_grad = False                  # weighting on shared params
            self.epoch = 0

            # DWA / GradNorm read [task_num, num_epochs] loss history
            self.train_loss_buffer = np.zeros([self.task_num, num_epochs])

            # Learnable params (UW: log-variance, GradNorm: loss_scale)
            self.init_param()

        # ── AbsWeighting required interface ──────────────────────────────────

        def get_share_params(self):
            """Shared backbone = Witra encoder only (not task heads)."""
            return self.wrapped_model.encoder.parameters()

        def zero_grad_share_params(self):
            for p in self.get_share_params():
                if p.grad is not None:
                    p.grad.zero_()

    # ─────────────────────────────────────────────────────────────────────────
    adapter = MTLAdapter()
    adapter.to(device)
    return adapter
