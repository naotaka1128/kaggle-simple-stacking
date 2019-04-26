import numpy as np
import pandas as pd


def save_commits(df, data, file_name):
    df_tmp = pd.DataFrame({
        "MachineIdentifier": df['MachineIdentifier'].values
    })
    df_tmp['HasDetections'] = data
    df_tmp.to_csv(file_name, index=False)
