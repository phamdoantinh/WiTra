import os
import re
import torch
import numpy as np
import pandas as pd
from typing import Optional, Tuple, Dict
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from . import data_process

class IndoorLocDataset(Dataset):
    def __init__(
            self,
            df: pd.DataFrame,
            wap_cols: list,
            is_train: bool = False,
            rssi_noise_std: float = 0.03,
            wap_drop_rate: float = 0.30,
            floor_mapping: Optional[Dict[int, int]] = None,
    ):
        self.df = df.reset_index(drop=True)
        self.wap_cols = wap_cols
        self.is_train = is_train
        self.rssi_noise_std = rssi_noise_std
        self.wap_drop_rate = wap_drop_rate

        # Detect coordinate and label columns
        if 'LONGITUDE' in self.df.columns and 'LATITUDE' in self.df.columns:
            self.x_col, self.y_col = 'LONGITUDE', 'LATITUDE'
            self.bld_col, self.flr_col = 'BUILDINGID', 'FLOOR'
            self.dataset_type = 'uji'
        elif 'Pos_x' in self.df.columns and 'Pos_y' in self.df.columns:
            self.x_col, self.y_col = 'Pos_x', 'Pos_y'
            # Tampere: uses 'Floor' + 'Building_ID' (already integer, no mapping needed)
            # UTS:     uses 'Floor_ID' + 'Building_ID' (needs floor mapping for [-3..12])
            if 'Floor' in self.df.columns:
                self.bld_col, self.flr_col = 'Building_ID', 'Floor'
                self.dataset_type = 'tampere'
            else:
                self.bld_col, self.flr_col = 'Building_ID', 'Floor_ID'
                self.dataset_type = 'uts'
                # Floor mapping for UTS: map sorted floors [-3..12] to [0..15]
                if floor_mapping is None:
                    unique_floors = sorted([-3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
                    self.floor_mapping = {f: i for i, f in enumerate(unique_floors)}
                else:
                    self.floor_mapping = floor_mapping
        else:
            raise KeyError("Coordinate columns not found in DataFrame. Must contain either LONGITUDE/LATITUDE or Pos_x/Pos_y.")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        features = torch.tensor(row[self.wap_cols].values.astype(np.float32), dtype=torch.float32)



        if self.is_train:
            # ── Aug 1: Gaussian noise on active APs only ──────────────────────
            active_mask = features > 0.0
            if active_mask.any():
                noise = torch.randn_like(features) * self.rssi_noise_std
                features = features + noise * active_mask.float()
                features = torch.clamp(features, 0.0, 1.0)

            # ── Aug 2: WAP random dropout ─────────────────────────────────────
            if active_mask.any() and self.wap_drop_rate > 0.0:
                drop_mask = (torch.rand_like(features) < self.wap_drop_rate) & active_mask
                features = features * (~drop_mask).float()

        coords = torch.tensor(row[[self.x_col, self.y_col]].values.astype(np.float32), dtype=torch.float32)
        
        if self.dataset_type == 'uts':
            # UTS has single string building 'New IT', map to 0
            building = torch.tensor(0, dtype=torch.long)
            # Map floor level [-3..12] to [0..15]
            raw_floor = int(row[self.flr_col])
            mapped_floor = self.floor_mapping.get(raw_floor, 0)
            floor = torch.tensor(mapped_floor, dtype=torch.long)
        else:
            # UJI and Tampere: building and floor already clean integers
            building = torch.tensor(int(row[self.bld_col]), dtype=torch.long)
            floor = torch.tensor(int(row[self.flr_col]), dtype=torch.long)

        return {
            'rssi features': features,
            'coords':   coords,
            'building': building,
            'floor':    floor,
        }



def get_dataloader(
        df_train: pd.DataFrame,
        df_test: pd.DataFrame,
        wap_cols: list,
        batch_size: int = 32,
        num_workers: int = 4,
        rssi_noise_std: float = 0.03,
        wap_drop_rate: float = 0.30,
):
    # Detect building and floor column names
    if 'BUILDINGID' in df_train.columns and 'FLOOR' in df_train.columns:
        bld_col, flr_col = 'BUILDINGID', 'FLOOR'
    elif 'Building_ID' in df_train.columns and 'Floor' in df_train.columns:
        # Tampere: Floor col (not Floor_ID)
        bld_col, flr_col = 'Building_ID', 'Floor'
    elif 'Building_ID' in df_train.columns and 'Floor_ID' in df_train.columns:
        bld_col, flr_col = 'Building_ID', 'Floor_ID'
    else:
        raise KeyError("Building or Floor columns not found in DataFrame.")

    # Stratified split so every building×floor is represented in both sets
    df_train_split, df_val_split = train_test_split(
        df_train,
        test_size=0.2,
        random_state=42,
        stratify=df_train[bld_col].astype(str) + '_' + df_train[flr_col].astype(str)
    )

    # Train: augmentation ON  |  Val / Test: augmentation OFF
    train_dataset = IndoorLocDataset(df=df_train_split, wap_cols=wap_cols,
                                     is_train=True,
                                     rssi_noise_std=rssi_noise_std,
                                     wap_drop_rate=wap_drop_rate)
    val_dataset   = IndoorLocDataset(df=df_val_split,   wap_cols=wap_cols, is_train=False)
    test_dataset  = IndoorLocDataset(df=df_test,         wap_cols=wap_cols, is_train=False)

    pin_mem = torch.cuda.is_available()
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,  num_workers=num_workers, pin_memory=pin_mem)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_mem)
    test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_mem)

    return train_loader, val_loader, test_loader
