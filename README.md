# WiTra: Vision Transformer for WiFi-based Indoor Positioning

WiTra is a Vision Transformer (ViT) architecture customized for multi-task WiFi-based indoor positioning. It jointly predicts the **Building** (classification), **Floor** (classification), and **Coordinates** (regression) of a device from WiFi Received Signal Strength Indicator (RSSI) fingerprints.

---

## Architecture Overview

1. **Vision Transformer Backbone**: Pre-processes 1D RSSI vectors into token embeddings and adds learnable positional encodings.
2. **Specialized Multi-Task Tokens**: Rather than a single class token, WiTra prepends three specialized learnable tokens:
   - `cls_token_building`: Captured representation for building classification.
   - `cls_token_floor`: Captured representation for floor classification.
   - `reg_token`: Captured representation for coordinate regression.
3. **Shared-Private Projection**: Separates task-specific features (cls/reg representations) into Shared and Private subspaces via a projection layer, applying difference losses (`l_diff` and `l_sp_diff`) to enforce orthogonality and prevent redundant feature learning.
4. **Soft-Argmax Coordinate Head**: Employs an expectation-based binning head (Soft-Argmax) instead of a direct linear regressor, producing smooth coordinate predictions constrained in $[0.0, 1.0]$.
5. **Differentiable Floor Conditioning**: The coordinate head accepts the coordinates regressor representation (`reg_token`) concatenated with the *soft probability distribution* of the floor classification head. By avoiding `detach()`, gradients from the coordinate regression head backpropagate into the floor classifier, acting as a spatial regularizer.
6. **Adaptive Multi-Task Weighting**: Integrated with the `LibMTL` library using Uncertainty Weighting (UW) to balance the classification and regression losses dynamically.
7. **Spatial Normalization over Union Bounding Box**: Normalizes target coordinates based on the spatial bounding box of the *union* of both train and test datasets, preventing out-of-bounds clipping by the Soft-Argmax head.
8. **Fingerprint Augmentation**: Supports K-Nearest Neighbors (KNN) fingerprint interpolation to augment sparse datasets (e.g., Tampere), mitigating overfitting.

---

## Project Structure

```text
WiTra/
├── dataprocess/
│   ├── data_loader.py       # Data loading & dataset loaders
│   └── data_process.py      # Normalization, augmentation, and dataset parser
├── model/
│   └── witra.py             # WiTra model (Embedding, Encoder, MultiHead, and Witra modules)
├── train/
│   ├── train.py             # Main training entry point
│   ├── train_core.py        # Core single-task/fixed trainer loop
│   ├── mtl_trainer.py       # Multi-task trainer with LibMTL adaptation
│   ├── mtl_adapter.py       # Bridge factory to AbsWeighting in LibMTL
│   └── mtl_losses.py        # Custom multi-task loss functions
└── README.md
```

---

## Installation & Setup

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/Greediex/WiTra.git
   cd WiTra
   ```

2. **Clone the `LibMTL` Dependency**:
   WiTra relies on the `LibMTL` framework for multi-task loss weighting. Clone the official library into the repository root:
   ```bash
   git clone https://github.com/median-research-group/LibMTL.git
   ```

3. **Install Dependencies**:
   Ensure you have PyTorch, NumPy, Pandas, and Tqdm installed:
   ```bash
   pip install torch numpy pandas tqdm
   ```

---

## Datasets Supported

- **UJIIndoorLoc** (`uji`)
- **UTSIndoorLoc** (`uts`)
- **Tampere** (`tampere`)

---

## Running Training

To run training, use the `train/train.py` script.

### Recommended Command (UJI Dataset)
For optimal convergence on UJIIndoorLoc, run:
```bash
python train/train.py \
  --dataset uji \
  --train-data uji_data/raw/trainingData.csv \
  --test-data uji_data/raw/validationData.csv \
  --norm-method mean_center \
  --batch-size 32 \
  --lr 1e-4
```

### Options
- `--dataset`: Choose between `uji`, `uts`, or `tampere`.
- `--train-data`: Path to train CSV file (required for `uji` and `uts`).
- `--test-data`: Path to test CSV file (required for `uji` and `uts`).
- `--tampere-dir`: Path to Tampere raw dataset directory (required for `tampere`).
- `--norm-method`: Normalization method (`normal`, `power`, or `mean_center`).
- `--lr`: Learning rate (default: `8e-4`).
- `--batch-size`: Batch size (default: `128`).
- `--epochs`: Number of training epochs (default: `300`).
- `--num-bins`: Number of bins for Soft-Argmax coordinate regression (default: `100`).
- `--w-diff`: Weight of the private subspace difference (orthogonality) loss (default: `0.01`).
- `--w-sp-diff`: Weight of the shared-private difference (orthogonality) loss (default: `0.01`).
- `--aug-n`: Fingerprint interpolation factor (automatically defaults to `8` for `tampere` to perform KNN augmentation, `0` otherwise).
- `--aug-k`: KNN neighbors for interpolation (default: `5`).
- `--floor-focal-gamma`: Focal Loss gamma for floor task (default: `2.0`).
- `--floor-acc-weight`: Weight for floor acc in composite checkpoint score (default: `10.0`).
