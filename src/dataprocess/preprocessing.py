import os
import numpy as np
import pandas as pd

_TAMPERE_Z_TO_FLOOR = {0.0: 0, 3.7: 1, 7.4: 2, 11.1: 3, 14.8: 4}

def replace_missing_values(df, mis=100, re=-110):
    # Sentinels belong to RSSI features, never coordinates or class metadata.
    out = df.copy()
    wap_cols = get_wap_cols(out)
    out[wap_cols] = out[wap_cols].replace(mis, re)
    return out

def get_wap_cols(df):
    """Return WAP feature column names. Supports UJI (WAP000) and Tampere (WAP_000) naming."""
    return [col for col in df.columns if col.startswith('WAP')]

def norm_mean_center_WAP(df, rel_unheard=-20.0, max_diff=50.0, re=-110):
    wap_cols = get_wap_cols(df)
    df_wap = df[wap_cols].values.astype(float)
    
    heard_mask = (df_wap != re)
    heard_counts = heard_mask.sum(axis=1, keepdims=True)
    row_means = np.full((len(df_wap), 1), -80.0)
    np.divide(np.where(heard_mask, df_wap, 0.0).sum(axis=1, keepdims=True),
              heard_counts, out=row_means, where=heard_counts > 0)
    
    mc_rssi = np.where(heard_mask, df_wap - row_means, rel_unheard)
    
    span = max_diff - rel_unheard
    norm_mc = (mc_rssi - rel_unheard) / span
    norm_mc = np.clip(norm_mc, 0.0, 1.0)
    
    df_out = df.copy()
    df_out[wap_cols] = norm_mc
    return df_out

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
