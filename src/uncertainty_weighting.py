"""Load the bundled LibMTL UW algorithm through its normal package import."""
import inspect
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parent
VENDOR = ROOT / 'LibMTL'
UW_SOURCE = VENDOR / 'LibMTL' / 'weighting' / 'UW.py'
if not UW_SOURCE.is_file():
    raise FileNotFoundError(f'Bundled LibMTL UW is missing: {UW_SOURCE}')
sys.path.insert(0, str(VENDOR))
from LibMTL.weighting import UW

if Path(inspect.getfile(UW)).resolve() != UW_SOURCE.resolve():
    raise RuntimeError('A different LibMTL is already imported. Use the bundled source in a fresh Python process.')

BACKEND = 'LibMTL.UW'
UPSTREAM_COMMIT = '4336804847eaa5e0b924b743d76beec7ac3fdc97'


class WiTraUW(UW):
    """LibMTL UW with the revision's explicitly active task list."""
    def __init__(self, active_tasks, device):
        super().__init__()
        if not active_tasks or len(set(active_tasks)) != len(active_tasks):
            raise ValueError('UW requires a nonempty unique active task list')
        self.task_name = list(active_tasks)
        self.task_num = len(self.task_name)
        self.device = torch.device(device)
        self.init_param()


def create_weighting(config, device):
    if config['uw_initial_log_scale'] != -0.5:
        raise ValueError('This runner supports the bundled LibMTL UW initialization only')
    return WiTraUW(config['active_tasks'], device)


def validate_checkpoint_weighting(checkpoint, config):
    scales = checkpoint.get('uw_log_scales')
    state = checkpoint.get('weighting_state_dict')
    if not isinstance(state, dict) or set(state) != {'loss_scale'}:
        raise ValueError('Checkpoint missing actual LibMTL UW state')
    value = state['loss_scale']
    if not isinstance(value, torch.Tensor) or value.shape != (len(config['active_tasks']),) or not torch.isfinite(value).all():
        raise ValueError('Invalid LibMTL UW scale state')
    if not isinstance(scales, torch.Tensor) or not torch.equal(value.cpu(), scales.cpu()):
        raise ValueError('Checkpoint UW scales disagree with LibMTL state')
