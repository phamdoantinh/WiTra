#!/usr/bin/env bash
# Train or evaluate WiTra.
#   bash run.sh train|test DATASET [SEED] [WIDTH]
#   DATASET: uji | uts | tampere   SEED: 42 (default), 43, 44
#   WIDTH: 160 (default), 64, 40
set -euo pipefail
cd "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"

if (( $# < 2 || $# > 4 )); then
  echo 'Usage: bash run.sh train|test uji|uts|tampere [seed] [width]' >&2
  exit 2
fi
action="$1"; dataset="$2"; seed="${3:-42}"; width="${4:-160}"
case "$dataset" in uji|uts|tampere) ;; *) echo "Unknown dataset: $dataset" >&2; exit 2 ;; esac
python_bin="${WITRA_PYTHON:-python}"
output_root="${WITRA_RUNS:-../runs}/d${width}"
device="${WITRA_DEVICE:-cuda}"
case "$action" in
  train)
    # Learning rate, batch size, depth, FFN ratio and precision come
    # from the paper defaults in run.py.
    exec "$python_bin" -B run.py train \
      --prepared "${WITRA_PREPARED_ROOT:-../prepared}/${dataset}_mean_center" \
      --output-root "$output_root" --seed "$seed" \
      --embed-dim "$width" --device "$device" --epochs 300 \
      --num-workers "${WITRA_WORKERS:-0}"
    ;;
  test)
    exec "$python_bin" -B run.py evaluate \
      --run-dir "$output_root/$dataset/MAIN/seed_$seed" --device "$device"
    ;;
  *) echo "Unknown action: $action (use train or test)" >&2; exit 2 ;;
esac
