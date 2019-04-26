import random
import numpy as np
import pandas as pd
import glob
from .preprocessing import scaler
from .set_pickle_files import set_pickle_files

def load_data(scaled=False, test_mode=False):
    df_train, df_test = load_data_base(scaled, test_mode)
    features = set(df_train.columns)
    additional_pickle_files = set_pickle_files()

    df_train_new = df_train
    df_test_new = df_test

    for file in additional_pickle_files:
        df_new_feature = pd.read_pickle(file)
        df_train_new = pd.merge(
            df_train_new, df_new_feature, on='MachineIdentifier', how='left')
        df_test_new = pd.merge(
            df_test_new, df_new_feature, on='MachineIdentifier', how='left')

    if test_mode:
        df_train_new = df_train_new.loc[:20000, :]
        df_test_new = df_test_new.loc[:20000, :]

    if scaled == False:
        return df_train_new, df_test_new
    else:
        new_features = set(df_train_new.columns) - features
        new_features.add('MachineIdentifier')  # Joinする時に利用する
        df_train_new_scaled = scaler(df_train_new.loc[:, new_features])
        df_test_new_scaled = scaler(df_test_new.loc[:, new_features])

        new_features.remove('MachineIdentifier')
        df_train_new.drop(new_features, axis=1, inplace=True)
        df_test_new.drop(new_features, axis=1, inplace=True)

        df_train = pd.merge(
            df_train_new, df_train_new_scaled, on='MachineIdentifier', how='left')
        df_test = pd.merge(
            df_test_new, df_test_new_scaled, on='MachineIdentifier', how='left')

        return df_train, df_test


def load_data_for_tester(scaled=False, test_mode=0):
    df_train, df_test = load_data_base(scaled, test_mode)
    additional_pickle_files = set_pickle_files()

    if test_mode == 1:
        df_train = df_train.loc[:20000, :]
        df_test = df_test.loc[:20000, :]

    unnecessary_columns = []

    df_train.drop(unnecessary_columns, axis=1, inplace=True)
    df_test.drop(unnecessary_columns, axis=1, inplace=True)

    if scaled == False:
        return df_train, df_test, additional_pickle_files
    else:
        return df_train, df_test, additional_pickle_files


def load_data_base(scaled=False, test_mode=False):
    if scaled:
        df_train = pd.read_pickle(
            './data/20190307_base/df_train_scaled.pickle')
        df_test = pd.read_pickle(
            './data/20190307_base/df_test_scaled.pickle')
    else:
        """
        df_train = pd.read_pickle(
            './data/original/df_train.pickle')
        df_test = pd.read_pickle(
            './data/original/df_test.pickle')
        """
        df_train = pd.read_pickle(
            './data/20190307_base/df_train.pickle')
        df_test = pd.read_pickle(
            './data/20190307_base/df_test.pickle')

    return df_train, df_test


def load_calculated_model_data(model_name, file_type):
    path = "./data/algos/{}_{}_*".format(model_name, file_type)
    files = glob.glob(path)
    if len(files) > 1:
        raise TypeError('Multi File Output is not supported yet.')
    elif len(files) == 1:
        return pd.read_pickle(files[0])
    else:
        return None
