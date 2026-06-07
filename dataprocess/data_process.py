import os
import numpy as np
import pandas as pd

def replace_missing_values(df, mis=100, re=-110):
    df = df.replace(mis, re)
    return df
def find_missing_cols(df, mis=100):
    missing_cols = [col for col in df.columns if (df[col] == mis).all()]
    return missing_cols
def drop_missing_cols(df, missing_cols):
    df = df.drop(columns=missing_cols)
    return df

def get_wap_cols(df):
    """Return WAP feature column names. Supports UJI (WAP000) and Tampere (WAP_000) naming."""
    return [col for col in df.columns if col.startswith('WAP')]

def norm_WAP(df):
    wap_cols = get_wap_cols(df)
    df_wap = df[wap_cols]
    df_wap_norm = (df_wap + 110) / 110
    df_out = df.copy()
    df_out[wap_cols] = df_wap_norm
    return df_out

def norm_power_WAP(df, b=2.71828):
    wap_cols = get_wap_cols(df)
    df_wap = df[wap_cols]
    out = pd.DataFrame(0.0, index=df.index, columns=df_wap.columns)
    out[df_wap >= 0] = 1.0
    mask = (df_wap >= -100) & (df_wap < 0)
    x_norm = df_wap[mask] / 100.0 + 1.0
    out[mask] = (b ** x_norm) / b
    df_out = df.copy()
    df_out[wap_cols] = out
    return df_out

def norm_mean_center_WAP(df, rel_unheard=-20.0, max_diff=50.0, re=-110):
    wap_cols = get_wap_cols(df)
    df_wap = df[wap_cols].values.astype(float)
    
    heard_mask = (df_wap != re)
    df_wap_nan = np.where(heard_mask, df_wap, np.nan)
    row_means = np.nanmean(df_wap_nan, axis=1, keepdims=True)
    row_means = np.nan_to_num(row_means, nan=-80.0)
    
    mc_rssi = np.where(heard_mask, df_wap - row_means, rel_unheard)
    
    span = max_diff - rel_unheard
    norm_mc = (mc_rssi - rel_unheard) / span
    norm_mc = np.clip(norm_mc, 0.0, 1.0)
    
    df_out = df.copy()
    df_out[wap_cols] = norm_mc
    return df_out

def count_wap_cols(df):
    return len(get_wap_cols(df))

def get_coord_cols(df):
    if 'LONGITUDE' in df.columns and 'LATITUDE' in df.columns:
        return 'LONGITUDE', 'LATITUDE'
    elif 'Pos_x' in df.columns and 'Pos_y' in df.columns:
        return 'Pos_x', 'Pos_y'
    else:
        raise KeyError("Coordinate columns not found. Need LONGITUDE/LATITUDE or Pos_x/Pos_y.")

def find_min_max_coord(df):
    x_col, y_col = get_coord_cols(df)
    min_x = df[x_col].min()
    max_x = df[x_col].max()
    min_y = df[y_col].min()
    max_y = df[y_col].max()
    range_x = max_x - min_x
    range_y = max_y - min_y
    max_range = max(range_x, range_y)
    return min_x, max_x, min_y, max_y, max_range

def transform_coord(df, min_x, min_y):
    df = df.copy()
    x_col, y_col = get_coord_cols(df)
    df[x_col] = df[x_col] - min_x
    df[y_col] = df[y_col] - min_y
    return df

def norm_coord(df, max_range):
    df = df.copy()
    x_col, y_col = get_coord_cols(df)
    df[x_col] = df[x_col] / max_range
    df[y_col] = df[y_col] / max_range
    return df


def process_data(
        df_train,
        df_test,
        mis=100,
        re=-110,
        norm_method="normal"
):
    missing_cols = find_missing_cols(df_train, mis)
    min_x, max_x, min_y, max_y, max_range = find_min_max_coord(df_train)

    df_train = drop_missing_cols(df_train, missing_cols)
    df_train = replace_missing_values(df_train, mis, re)
    
    # Prevent data leakage: drop duplicate fingerprints in train set
    wap_cols = [col for col in df_train.columns if col.startswith('WAP')]
    df_train = df_train.drop_duplicates(subset=wap_cols)
    
    df_train = transform_coord(df_train, min_x, min_y)
    _,_,_,_,range = find_min_max_coord(df_train)
    df_train = norm_coord(df_train, range)
    
    if norm_method == "normal":
        df_train = norm_WAP(df_train)
    elif norm_method == "power":
        df_train = norm_power_WAP(df_train)
    elif norm_method == "mean_center":
        df_train = norm_mean_center_WAP(df_train, re=re)
    else:
        raise ValueError(f"Unknown norm_method: {norm_method}")

    df_test = drop_missing_cols(df_test, missing_cols)
    df_test = replace_missing_values(df_test, mis, re)
    df_test = transform_coord(df_test, min_x, min_y)
    # Test coords are NOT normalized - keep in transformed space [0, max_range]
    
    if norm_method == "normal":
        df_test = norm_WAP(df_test)
    elif norm_method == "power":
        df_test = norm_power_WAP(df_test)
    elif norm_method == "mean_center":
        df_test = norm_mean_center_WAP(df_test, re=re)
    else:
        raise ValueError(f"Unknown norm_method: {norm_method}")

    return df_train, df_test, range


# ---------------------------------------------------------------------------
# Fingerprint interpolation augmentation
# ---------------------------------------------------------------------------
# Generates synthetic training samples by linearly interpolating between
# K nearest neighbor reference points on the SAME floor.
# RSSI_new = α·RSSI_A + (1-α)·RSSI_B  |  coord_new = α·coord_A + (1-α)·coord_B
# Only heard APs (value > 0 after norm) are interpolated; unheard stays 0.
# This is the primary technique papers use to reach ~7m on Tampere's 697 samples.
# ---------------------------------------------------------------------------

def augment_fingerprints(
    df: pd.DataFrame,
    wap_cols: list,
    n_aug_per_sample: int = 8,
    k_neighbors: int = 5,
    same_floor_only: bool = True,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Linear interpolation fingerprint augmentation.

    Args:
        df               : processed training DataFrame (WAP cols already normalized)
        wap_cols         : list of WAP column names in df
        n_aug_per_sample : synthetic samples to generate per reference point
        k_neighbors      : # nearest neighbors to interpolate with (by coord distance)
        same_floor_only  : only interpolate within same floor (strongly recommended)
        seed             : random seed for reproducibility

    Returns:
        df_augmented : original df concatenated with synthetic samples (same schema)
    """
    from sklearn.neighbors import NearestNeighbors

    rng = np.random.default_rng(seed)

    x_col, y_col = get_coord_cols(df)

    # Detect floor column
    if 'Floor' in df.columns:
        flr_col = 'Floor'
    elif 'FLOOR' in df.columns:
        flr_col = 'FLOOR'
    else:
        flr_col = None  # no floor column — interpolate across all

    rssi_vals  = df[wap_cols].values.astype(np.float32)   # (N, n_waps)
    coord_vals = df[[x_col, y_col]].values.astype(np.float32)  # (N, 2)
    floors     = df[flr_col].values if flr_col else np.zeros(len(df), dtype=int)

    # Non-WAP columns (floor, building, etc.) — carry over from anchor sample
    meta_cols = [c for c in df.columns if c not in wap_cols and c not in (x_col, y_col)]

    synthetic_rows = []

    unique_floors = np.unique(floors)
    for flr in unique_floors:
        floor_mask = floors == flr
        idx_floor  = np.where(floor_mask)[0]

        if len(idx_floor) < 2:
            continue  # can't interpolate with <2 points

        coords_flr = coord_vals[idx_floor]
        rssi_flr   = rssi_vals[idx_floor]

        # KNN on 2D coordinates within this floor
        k = min(k_neighbors, len(idx_floor) - 1)
        nn = NearestNeighbors(n_neighbors=k + 1, algorithm='ball_tree')
        nn.fit(coords_flr)
        _, nbr_indices = nn.kneighbors(coords_flr)  # (M, k+1); col 0 = self

        for i, anchor_local in enumerate(range(len(idx_floor))):
            # Neighbors excluding self
            neighbor_locals = nbr_indices[i, 1:]  # (k,)

            for _ in range(n_aug_per_sample):
                # Pick a random neighbor
                j = int(rng.choice(neighbor_locals))
                alpha = float(rng.uniform(0.1, 0.9))

                # Interpolate RSSI: only if BOTH are heard; else keep anchor's value
                rssi_a = rssi_flr[anchor_local]
                rssi_b = rssi_flr[j]
                heard_both = (rssi_a > 0) & (rssi_b > 0)
                heard_any  = (rssi_a > 0) | (rssi_b > 0)
                rssi_new = np.where(heard_both,
                                    alpha * rssi_a + (1 - alpha) * rssi_b,
                                    np.where(heard_any,
                                             np.maximum(rssi_a, rssi_b),
                                             0.0)).astype(np.float32)

                # Interpolate coordinates
                coord_new = alpha * coord_vals[idx_floor[anchor_local]] + \
                            (1 - alpha) * coord_vals[idx_floor[j]]

                # Build row dict
                row = {wc: float(rssi_new[wi]) for wi, wc in enumerate(wap_cols)}
                row[x_col] = float(coord_new[0])
                row[y_col] = float(coord_new[1])

                # Copy meta columns from anchor
                anchor_global = idx_floor[anchor_local]
                for mc in meta_cols:
                    row[mc] = df.iloc[anchor_global][mc]

                synthetic_rows.append(row)

    df_synthetic = pd.DataFrame(synthetic_rows, columns=df.columns)
    df_aug = pd.concat([df, df_synthetic], ignore_index=True)
    return df_aug


# ---------------------------------------------------------------------------
# Tampere University (TUT) dataset loader
# ---------------------------------------------------------------------------
# Dataset structure (raw/ subfolder):
#   Training_rss.csv  / Test_rss.csv  — no header, 992 AP columns, sentinel=100
#   Training_coordinates.csv / Test_coordinates.csv — no header, cols: x, y, z (metres)
# z encodes floor height: 0.0→F0, 3.7→F1, 7.4→F2, 11.1→F3, 14.8→F4 (single building)
#
# Output DataFrame schema (compatible with IndoorLocDataset):
#   WAP_000 … WAP_991  : raw RSS (sentinel still 100, preprocessing applied later)
#   Pos_x, Pos_y       : local coordinates in metres
#   Floor              : integer 0-4
#   Building_ID        : always 0
# ---------------------------------------------------------------------------

_TAMPERE_Z_TO_FLOOR = {0.0: 0, 3.7: 1, 7.4: 2, 11.1: 3, 14.8: 4}


def load_tampere(data_dir: str):
    """
    Load Tampere (TUT) indoor localization dataset.

    Args:
        data_dir: path to the tampere_data folder that contains raw/

    Returns:
        df_train, df_test : DataFrames with WAP_xxx/Pos_x/Pos_y/Floor/Building_ID
        wap_cols          : list of WAP column names
    """
    raw_dir = os.path.join(data_dir, 'raw')

    rss_tr   = pd.read_csv(os.path.join(raw_dir, 'Training_rss.csv'),         header=None)
    coord_tr = pd.read_csv(os.path.join(raw_dir, 'Training_coordinates.csv'), header=None)
    rss_te   = pd.read_csv(os.path.join(raw_dir, 'Test_rss.csv'),             header=None)
    coord_te = pd.read_csv(os.path.join(raw_dir, 'Test_coordinates.csv'),     header=None)

    n_aps    = rss_tr.shape[1]  # 992
    wap_cols = [f'WAP_{i:03d}' for i in range(n_aps)]

    def _build_df(rss, coord):
        df = rss.copy()
        df.columns = wap_cols
        df['Pos_x']       = coord[0].values
        df['Pos_y']       = coord[1].values
        df['Floor']       = coord[2].map(_TAMPERE_Z_TO_FLOOR).astype(int)
        df['Building_ID'] = 0
        return df

    df_train = _build_df(rss_tr, coord_tr)
    df_test  = _build_df(rss_te, coord_te)

    return df_train, df_test, wap_cols