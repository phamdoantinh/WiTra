#!/usr/bin/env bash
set -euo pipefail
cd "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
dataset="${1:?Usage: prepare_data.sh uji|uts|tampere [EXTERNAL_DATA_ROOT]}"
data_root="${2:-${WITRA_DATA_ROOT:-../data/indoorloc}}"
if [[ ! -d "$data_root" ]]; then
  echo 'Raw data is external. Set WITRA_DATA_ROOT or pass EXTERNAL_DATA_ROOT as argument 2.' >&2
  exit 2
fi
exec "${WITRA_PYTHON:-.venv/bin/python}" -B prepare_data.py --dataset "$dataset" --data-root "$data_root" --output-dir "${WITRA_PREPARED_ROOT:-../prepared}/${dataset}_mean_center"
