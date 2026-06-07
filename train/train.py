"""
Unified WiFi Indoor Positioning (UJI, UTS, Tampere) training entry point.
Uses LibMTL-based MTLTrainer (UW-only).

Usage Examples
--------------
UJI:
python train/train.py --dataset uji --train-data uji_data/raw/trainingData.csv --test-data uji_data/raw/validationData.csv

UTS:
python train/train.py --dataset uts --train-data UTSIndoorLoc-dataset/uts_data/raw/UTS_training.csv --test-data UTSIndoorLoc-dataset/uts_data/raw/UTS_test.csv

Tampere:
python train/train.py --dataset tampere --tampere-dir tampere_data
"""
import os, sys, json, logging
import torch, numpy as np, pandas as pd
from pathlib import Path
from datetime import datetime

# Enable TensorFloat-32 (TF32) tensor cores
if torch.cuda.is_available():
    torch.set_float32_matmul_precision('high')

sys.path.insert(0, str(Path(__file__).parent.parent))

from dataprocess.data_loader import get_dataloader
from dataprocess.data_process import load_tampere, process_data, count_wap_cols, augment_fingerprints
from model.witra import Witra
try:
    from train.mtl_trainer import MTLTrainer
except (ImportError, ModuleNotFoundError):
    from mtl_trainer import MTLTrainer


def set_seed(seed: int = 42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True


def setup_logger(save_dir: str, dataset: str):
    os.makedirs(save_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(save_dir, f"{dataset}_train_{ts}.log")
    logger = logging.getLogger("mtl_training")
    logger.setLevel(logging.INFO)
    logger.handlers = []
    for h in [logging.FileHandler(log_file, encoding="utf-8"),
               logging.StreamHandler(sys.stdout)]:
        h.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(h)
    return logger, log_file


def main(args):
    dataset = args.dataset.lower()

    # ── Hyperparameters ──────────────────────────────────────────
    embed_dim = args.embed_dim
    depth = args.depth
    num_heads = args.num_heads
    dropout = args.dropout
    batch_size = args.batch_size
    lr = args.lr
    rssi_noise_std = args.rssi_noise_std
    wap_drop_rate = args.wap_drop_rate
    
    norm_method = args.norm_method
    aug_n = args.aug_n if args.aug_n is not None else (8 if dataset == "tampere" else 0)
    aug_k = args.aug_k
    
    save_dir = args.save_dir or f"./checkpoints/{dataset}"
    
    logger, log_file = setup_logger(save_dir, dataset)
    set_seed(42)

    logger.info("=" * 60)
    logger.info(f"Indoor Positioning MTL — Dataset: {dataset.upper()} | strategy=UW (LibMTL)")
    logger.info("=" * 60)
    logger.info(f"epochs={args.epochs}  lr={lr}  batch={batch_size}  device={args.device}")
    logger.info(f"normalization={norm_method}  save_dir={save_dir}")
    logger.info(f"log: {log_file}")

    # ── Data Loading ─────────────────────────────────────────────────────────
    if dataset == "tampere":
        if not args.tampere_dir:
            raise ValueError("--tampere-dir is required when using the tampere dataset.")
        logger.info(f"Loading Tampere data from: {args.tampere_dir}")
        df_train, df_test, wap_cols = load_tampere(args.tampere_dir)
    else:
        if not args.train_data or not args.test_data:
            raise ValueError("--train-data and --test-data are required for UJI/UTS datasets.")
        df_train = pd.read_csv(args.train_data)
        df_test  = pd.read_csv(args.test_data)
        
    logger.info(f"raw: train={len(df_train)}  test={len(df_test)}")

    # ── Preprocessing ────────────────────────────────────────────────────────
    df_train, df_test, max_range = process_data(
        df_train=df_train, df_test=df_test, mis=100, re=-110, norm_method=norm_method
    )
    wap_cols = [col for col in df_train.columns if col.startswith("WAP")]
    logger.info(f"max_range={max_range:.2f}m  active_APs={len(wap_cols)}")

    # ── Fingerprint Augmentation (Tampere KNN interpolation) ──────────────────
    if dataset == "tampere" and aug_n > 0:
        n_before = len(df_train)
        df_train = augment_fingerprints(
            df=df_train, wap_cols=wap_cols,
            n_aug_per_sample=aug_n,
            k_neighbors=aug_k,
            same_floor_only=True,
            seed=42,
        )
        logger.info(f"Augmented train: {n_before} → {len(df_train)} samples (n_aug={aug_n}, k={aug_k})")

    # ── Dataloaders ──────────────────────────────────────────────────────────
    train_loader, val_loader, test_loader = get_dataloader(
        df_train=df_train, df_test=df_test, wap_cols=wap_cols,
        batch_size=batch_size, num_workers=args.num_workers,
        rssi_noise_std=rssi_noise_std,
        wap_drop_rate=wap_drop_rate,
    )
    logger.info(f"batches: train={len(train_loader)} val={len(val_loader)} test={len(test_loader)}")

    # ── Model Dimensions ─────────────────────────────────────────────────────
    if dataset == "uji":
        num_buildings = int(df_train['BUILDINGID'].max() + 1)
        num_floors = int(df_train['FLOOR'].max() + 1)
    elif dataset == "uts":
        num_buildings = 1
        num_floors = 16
    elif dataset == "tampere":
        num_buildings = int(df_train['Building_ID'].max() + 1)
        num_floors = int(df_train['Floor'].max() + 1)
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")

    model = Witra(
        encoder_cfg={
            "input_dim": len(wap_cols),
            "embed_dim": embed_dim,
            "depth":     depth,
            "block_cfg": {
                "mlp_ratio": 4,
                "attn_cfg": {
                    "num_heads":     num_heads,
                    "qkv_bias":      False,
                    "qk_norm":       True,
                    "porj_bias":     True,
                    "attn_drop":     0.0,
                    "proj_drop":     0.0,
                },
            },
            "learn_pos": True,
            "dropout":   dropout,
        },
        multihead_cfg={
            "dim":           embed_dim,
            "num_buildings": num_buildings,
            "num_floors":    num_floors,
            "dropout":       dropout,
        },
    )
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"model params={n_params:,}")

    # ── Trainer ───────────────────────────────────────────────────────────────
    trainer = MTLTrainer(
        model                 = model,
        device                = args.device,
        logger                = logger,
        norm_params           = max_range,
        mtl_strategy          = "UW",
        num_epochs            = args.epochs,
        mtl_kwargs            = {},
        floor_focal_gamma     = args.floor_focal_gamma,
        floor_acc_weight      = args.floor_acc_weight,
    )
    trainer.setup_training(
        learning_rate = lr,
        weight_decay  = 0.01,
    )
    if args.scheduler != "none":
        trainer.setup_scheduler(
            scheduler_type = args.scheduler,
            num_epochs     = args.epochs,
        )

    # ── Train ─────────────────────────────────────────────────────────────────
    history = trainer.fit(
        train_loader = train_loader,
        val_loader   = val_loader,
        test_loader  = test_loader,
        num_epochs   = args.epochs,
        save_dir     = save_dir,
        save_every   = 10,
        test_every   = 5,
    )

    # ── Test ──────────────────────────────────────────────────────────────────
    best_path = os.path.join(save_dir, "best_model.pth")
    if os.path.exists(best_path):
        logger.info(f"\nLoading best model checkpoint from {best_path} for final test evaluation...")
        trainer.load_checkpoint(best_path, load_optimizer=False)
    else:
        logger.warning(f"\nWarning: best_model.pth not found at {best_path}. Evaluating on final epoch weights.")
        
    test_results = trainer.test(test_loader)

    # ── Save ─────────────────────────────────────────────────────────────────
    trainer.save_history(os.path.join(save_dir, "history.json"))
    
    executed_cfg = {
        "dataset": dataset,
        "embed_dim": embed_dim,
        "depth": depth,
        "num_heads": num_heads,
        "dropout": dropout,
        "epochs": args.epochs,
        "batch_size": batch_size,
        "lr": lr,
        "rssi_noise_std": rssi_noise_std,
        "wap_drop_rate": wap_drop_rate,
        "norm_method": norm_method,
        "aug_n": aug_n,
        "aug_k": aug_k,
        "save_dir": save_dir,
    }
    with open(os.path.join(save_dir, "config.json"), "w") as f:
        json.dump(executed_cfg, f, indent=2, default=str)

    logger.info("\n" + "=" * 60)
    logger.info("RESULTS")
    logger.info("=" * 60)
    logger.info(f"best val coord error : {min(history['val_coord_error']):.2f}m")
    logger.info(f"best val floor acc   : {max(history['val_floor_acc']):.2f}%")
    logger.info(f"test error           : {test_results['coord_error_mean']:.2f}m  (med={test_results['coord_error_median']:.2f}m)")
    logger.info(f"building acc         : {test_results['building_acc']:.2f}%")
    logger.info(f"floor acc            : {test_results['floor_acc']:.2f}%")

    return trainer, history, test_results


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Unified WiFi Indoor Positioning (UJI, UTS, Tampere) - UW Strategy")

    # Dataset Choice
    p.add_argument("--dataset", required=True, choices=["uji", "uts", "tampere"],
                   help="Select the dataset type (uji | uts | tampere)")

    # Data paths (dynamic validation in main)
    p.add_argument("--train-data", default=None, help="Path to UJI/UTS train CSV")
    p.add_argument("--test-data",  default=None, help="Path to UJI/UTS test CSV")
    p.add_argument("--tampere-dir", default=None, help="Path to Tampere raw folder")

    # Model (same defaults as run.py)
    p.add_argument("--embed-dim",  type=int,   default=256,  help="Embedding dim")
    p.add_argument("--depth",      type=int,   default=6,    help="Transformer depth (was 4)")
    p.add_argument("--num-heads",  type=int,   default=8,    help="Attention heads")
    p.add_argument("--dropout",    type=float, default=0.1,  help="Dropout")

    # Training
    p.add_argument("--epochs",      type=int,   default=300,          help="Epochs (300 for augmentation)")
    p.add_argument("--batch-size",  type=int,   default=128,          help="Batch size")
    p.add_argument("--lr",          type=float, default=8e-4,         help="Learning rate")
    p.add_argument("--scheduler",   type=str,   default="cosine",     help="cosine|step|none")
    p.add_argument("--device",      type=str,   default="cuda",       help="cuda|cpu")
    p.add_argument("--save-dir",    type=str,   default=None)
    p.add_argument("--num-workers", type=int,   default=4)

    # Augmentation
    p.add_argument("--rssi-noise-std", type=float, default=0.03,  help="Std of Gaussian noise on active APs (0=off)")
    p.add_argument("--wap-drop-rate",  type=float, default=0.30,  help="Fraction of active WAPs to zero out (0=off)")
    p.add_argument("--norm-method",    type=str,   default="normal", choices=["normal", "power", "mean_center"],
                   help="Normalization method for WAP features")
    p.add_argument("--aug-n",          type=int,   default=None,  help="Fingerprint interpolation factor")
    p.add_argument("--aug-k",          type=int,   default=5,     help="KNN neighbors for interpolation")

    # Floor improvement
    p.add_argument("--floor-focal-gamma",     type=float, default=2.0,
                   help="Focal Loss gamma for floor task (0=plain CE, 2=default focal)")
    p.add_argument("--floor-acc-weight",      type=float, default=10.0,
                   help="Weight for floor acc in composite checkpoint score (higher=prioritize floor)")

    args = p.parse_args()

    try:
        main(args)
    except KeyboardInterrupt:
        print("\nInterrupted")
    except Exception as e:
        import traceback
        print(f"\nERROR: {e}")
        traceback.print_exc()
        sys.exit(1)
