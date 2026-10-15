# WiTra: A Compact Transformer for Hierarchical WiFi Indoor Localization

WiTra jointly predicts building, floor and 2D coordinates from a WiFi RSSI
fingerprint. Each retained access point (AP) becomes one token; three
task-specific readout tokens pass through a shared Transformer encoder, and the
predicted floor probabilities condition an expectation-based coordinate
decoder. During training, a floor-head update correction constrains the
first-order floor-loss contribution of each Floor Head step relative to a
floor-supervised reference step.

This repository contains the code for the revised manuscript submitted to
*Results in Engineering*: data preparation, the model, training, evaluation and
all configurations reported in the paper.

![WiTra architecture](assets/witra_architecture.png)

## Repository layout

```text
src/
  run.py                  command-line entry point (train / evaluate / aggregate)
  trainer.py              training, validation-selected evaluation, aggregation
  floor_protection.py     floor-head update correction (Eq. 13, Algorithm 1)
  uncertainty_weighting.py LibMTL uncertainty weighting (bundled, MIT license)
  prepare_data.py         splits, training-fitted preprocessing, TUT augmentation
  model/witra.py          embedding, encoder, task heads
  dataprocess/            RSSI normalization and dataset readers
  scripts/                setup, data preparation and run launchers
```

## Setup

Python 3.12 with PyTorch 2.8 (CUDA). BF16 training needs a GPU with BF16 support.

```bash
bash src/scripts/setup.sh                       # creates src/.venv
export WITRA_PYTHON="$PWD/src/.venv/bin/python"  # or the python of an existing environment
export WITRA_DEVICE=cuda:0
```

## Data

Raw datasets are not redistributed. Download UJIIndoorLoc, UTSIndoorLoc and
Tampere (TUT2017) and arrange them as:

```text
indoorloc/
  uji_data/raw/{trainingData.csv,validationData.csv}
  UTSIndoorLoc-dataset/uts_data/raw/{UTS_training.csv,UTS_test.csv}
  tampere_data/raw/{Training_rss.csv,Training_coordinates.csv,Test_rss.csv,Test_coordinates.csv}
```

Prepare each dataset once; all training seeds share these partitions:

```bash
export WITRA_DATA_ROOT=/absolute/path/to/indoorloc
export WITRA_PREPARED_ROOT="$PWD/prepared"
bash src/scripts/prepare_data.sh uji
bash src/scripts/prepare_data.sh uts
bash src/scripts/prepare_data.sh tampere
```

The official UJIIndoorLoc validation file is the test set. Internal validation
is fold 0 of a five-fold stratified group split of the official training data
(split seed 42; identical fingerprints stay in one fold). AP filtering,
coordinate bounds and the mean-centred RSSI transform are fitted on the
internal training partition. For TUT, KNN augmentation (k = 5, eight samples
per fingerprint) is applied to the internal training partition only.

## Reproducing the paper

`run.py` defaults to the paper's settings: depth 2, width 160, FFN ratio 1,
eight heads, 100 bins per axis, AdamW (weight decay 0.01), cosine schedule to
zero, 300 epochs, BF16 with SDPA, learning rate 8e-4 / 1e-4 / 1e-4 and batch
size 32 / 32 / 16 for UJIIndoorLoc / UTS / TUT.

```bash
export WITRA_RUNS="$PWD/runs"
bash src/scripts/run.sh train uji 42            # dataset, seed [width]
bash src/scripts/run.sh test  uji 42
bash src/scripts/run.sh train uts 43 40         # embedding width 40
```

Repeat for seeds 42, 43 and 44 and for `uji`, `uts` and `tampere`. Widths 64
and 40 give the smaller models of Table 7. Mean and sample standard deviation
across seeds:

```bash
cd src
python run.py aggregate --output-root "$WITRA_RUNS/d160" \
  --datasets uji uts tampere --seeds 42 43 44 --output "$WITRA_RUNS/d160/summary.json"
```

## Outputs

```text
runs/d160/uji/MAIN/seed_42/
  manifest.json     configuration, software, data and source hashes, selected epoch
  history.json      per-epoch training losses and validation metrics
  best.pt           validation-selected checkpoint
  metrics.json      test metrics (after evaluate)
  predictions.csv   per-sample predictions and errors (after evaluate)
```

Mean 2D error is the Euclidean distance between predicted and true horizontal
coordinates in metres. Runs are never overwritten. Evaluation verifies that the
current source files match the hashes recorded at training time; evaluate a run
with the code revision that trained it.

## License

The bundled LibMTL files keep their MIT license (`src/LibMTL/LICENSE`).
