# WiTra: Vision Transformer for WiFi-based Indoor Positioning

WiTra is a Vision Transformer (ViT) architecture customized for multi-task WiFi-based indoor positioning. It jointly predicts the **Building** (classification), **Floor** (classification), and **Coordinates** (regression) of a device from WiFi Received Signal Strength Indicator (RSSI) fingerprints.

---

## Architecture Overview

1. **Vision Transformer Backbone**: Pre-processes 1D RSSI vectors into token embeddings and adds learnable positional encodings.
2. **Specialized Multi-Task Tokens**: Rather than a single class token, WiTra prepends three specialized learnable tokens:
   - `cls_token_building`: Captured representation for building classification.
   - `cls_token_floor`: Captured representation for floor classification.
   - `reg_token`: Captured representation for coordinate regression.
3. **Differentiable Floor Conditioning**: The coordinate head accepts the coordinates regressor representation (`reg_token`) concatenated with the *soft probability distribution* of the floor classification head. By avoiding `detach()`, gradients from the coordinate regression head backpropagate into the floor classifier, acting as a spatial regularizer.
4. **Adaptive Multi-Task Weighting**: Integrated with the `LibMTL` library, supporting various loss weighting strategies such as Uncertainty Weighting (UW), Equal Weighting (EW), GradNorm, PCGrad, CAGrad, MGDA, FAMO, and GradVac.

---

## Project Structure

```text
gitupload/
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

### Recommended Hyperparameters for Optimal Performance
For optimal convergence (e.g. achieving `~3.0m` validation error on UJIIndoorLoc instead of `7.0m`+), it is highly recommended to use a batch size of `32` and learning rate of `1e-4` (or `8e-4` depending on the scheduler/normalization context):

```bash
python train/train.py \
  --dataset uji \
  --train-data /path/to/trainingData.csv \
  --test-data /path/to/validationData.csv \
  --norm-method mean_center \
  --batch-size 32 \
  --lr 1e-4 \
  --weighting UW
```

### Options
- `--dataset`: Choose between `uji`, `uts`, or `tampere`.
- `--norm-method`: Normalization method (`normal`, `power`, or `mean_center`).
- `--weighting`: Loss weighting strategy (default: `UW`). Supported options: `EW`, `UW`, `GradNorm`, `DWA`, `PCGrad`, `CAGrad`, `MGDA`, `FAMO`, `GradVac`.
- `--floor-acc-weight`: Scaling factor for floor classification accuracy in loss calculations.
- `--floor-focal-gamma`: Focal loss gamma parameter for floor prediction.
