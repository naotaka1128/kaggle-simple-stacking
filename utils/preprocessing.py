import re
import numpy as np
import pandas as pd
from functools import partial
from sklearn.preprocessing import StandardScaler
from utils.parallelize_dataframe import parallelize_dataframe

def replace_values(seq, seq_without_inf_nan):
    """
    異常値: clipping
    inf / -inf: clipping後の max/min ± mean*10
    nan: clipping後のmin - mean*5

    for i in range(len(seq)):
        if seq[i] == np.inf:
            seq[i] = df_max + df_std
        elif seq[i] == -np.inf:
            seq[i] = df_min - df_std
        elif np.isnan(seq[i]):
            seq[i] = df_min - df_std / 2
        else:
            seq[i] = seq_without_inf_nan[i]
    """
    if np.isinf(seq_without_inf_nan.sum()):
        if seq_without_inf_nan.dtype == 'float32':
            seq_without_inf_nan = seq_without_inf_nan.astype('float64')
        else:
            seq_without_inf_nan = seq_without_inf_nan.astype('float32')

    seq_mean = seq_without_inf_nan.mean()
    seq_max = seq_without_inf_nan.max()
    seq_min = seq_without_inf_nan.min()
    seq_std = seq_without_inf_nan.std()

    seq_without_inf_nan[seq == np.inf]  = seq_max + seq_std
    seq_without_inf_nan[seq == -np.inf] = seq_max - seq_std
    seq_without_inf_nan[seq.isnull()] = seq_min - seq_std / 2

    return seq_without_inf_nan.values


def scaler(df_target, num_cols):
    sc = StandardScaler()

    # For Test
    #exec_standard_scale(num_cols, sc, df_target.loc[:, num_cols[0]])

    f = partial(exec_standard_scale, num_cols, sc)
    df = parallelize_dataframe(df_target, f, columnwise=True)
    return df

def exec_standard_scale(num_cols, sc, seq):
    if seq.name not in num_cols:
        return seq

    seq_without_inf_nan = seq.replace([np.inf, -np.inf], np.nan).fillna(0)
    lowerbound, upperbound = np.percentile(seq_without_inf_nan, [0, 99])
    seq_without_inf_nan = np.clip(seq_without_inf_nan, lowerbound, upperbound)

    series = pd.Series(
        sc.fit_transform(
            replace_values(seq, seq_without_inf_nan).reshape(-1,1)
        ).reshape(-1,)
    )
    series.name = seq.name

    return series
