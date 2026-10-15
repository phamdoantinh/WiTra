#!/usr/bin/env bash
set -euo pipefail
cd "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ ! -d .venv ]]; then
  "${WITRA_PYTHON_BASE:-python3.12}" -m venv .venv
fi
.venv/bin/python -m pip install -r requirements-data.txt
.venv/bin/python -m pip install torch==2.8.0 --index-url "${WITRA_TORCH_INDEX:-https://download.pytorch.org/whl/cu128}"
.venv/bin/python -c "import torch; print(torch.__version__, torch.version.cuda); assert torch.cuda.is_available(), 'CUDA unavailable'; print(torch.cuda.get_device_name(0))"
.venv/bin/python -c "import uncertainty_weighting as b; print('MTL_BACKEND=' + b.BACKEND); print('UW_SOURCE=' + str(b.UW_SOURCE))"
echo 'SETUP_CUDA=PASS'
