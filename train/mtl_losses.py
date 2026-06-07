"""
IndoorLoc task-specific loss functions wrapping LibMTL's AbsLoss interface.

Tasks:
  - coord    : coordinate regression  → MSE
  - building : building classification → CrossEntropy
  - floor    : floor classification   → CrossEntropy

No silent fallback: unsupported args raise ValueError immediately.
"""
from pathlib import Path
import sys

# ── add LibMTL package to path ──────────────────────────────────────────────
_LIBMTL_ROOT = Path(__file__).parent.parent / "LibMTL"
if not _LIBMTL_ROOT.exists():
    raise FileNotFoundError(
        f"LibMTL not found at {_LIBMTL_ROOT}. "
        "Clone it: git clone https://github.com/median-research-group/LibMTL.git"
    )
if str(_LIBMTL_ROOT) not in sys.path:
    sys.path.insert(0, str(_LIBMTL_ROOT))

import torch
import torch.nn.functional as F
from LibMTL.loss import AbsLoss


class CoordLoss(AbsLoss):
    """
    Coordinate regression loss (MSE only).

    AbsLoss interface:
        _update_loss(pred, gt) → loss tensor  + records for averaging
        _average_loss()        → batch-weighted mean over epoch
        _reinit()              → clear accumulators
    """

    def compute_loss(self, pred, gt):
        return F.mse_loss(pred, gt)


class BuildingLoss(AbsLoss):
    """Building-ID classification loss (CrossEntropy)."""

    def compute_loss(self, pred, gt):
        return F.cross_entropy(pred, gt)


class FloorLoss(AbsLoss):
    """Floor-number classification loss.

    Uses Focal Loss (Lin et al. 2017) to focus training on hard-to-classify
    floor samples (border cases between adjacent floors).

    Args:
        gamma (float): Focal loss exponent. 0 = standard CE. Default 2.
    """

    def __init__(self, gamma: float = 2.0):
        super().__init__()
        self.gamma           = gamma

    def compute_loss(self, pred, gt):
        # 1. Unsmoothed CE to get the true class probability p_t
        ce_raw = F.cross_entropy(pred, gt, reduction='none')
        p_t = torch.exp(-ce_raw)
        
        # 2. Apply focal weight calculated from pure p_t to the loss
        focal_weight = (1.0 - p_t) ** self.gamma
        return (focal_weight * ce_raw).mean()
