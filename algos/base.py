import re
import random
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import roc_auc_score
from utils.load_data import load_data_for_tester, load_calculated_model_data
from utils.reduce_mem_usage import reduce_mem_usage

class BaseModel:
    def __init__(self, param_pattern=0, n_stacks=1, n_splits=5):
        self.param_pattern = param_pattern
        self.n_stacks = n_stacks
        self.n_splits = n_splits

    def fit_predict(self, test_mode=False):
        print('################### Start {}'.format(self.name))
        oof, pred = self._load_trained_data()
        if pred is not None:
            print('### File Exist. Use it.')
            oof.drop('machine_id', axis=1, inplace=True)
            pred.drop('machine_id', axis=1, inplace=True)
            return oof.iloc[:,0].values, pred.iloc[:,0].values

        df_train, df_test = self._load_data(test_mode)
        X_train = df_train.drop(
            ['HasDetections', 'MachineIdentifier', 'machine_id',
             'AvSigVersion_1', 'test_probability'], axis=1)
        X_test = df_test.drop(
            ['MachineIdentifier', 'machine_id',
             'AvSigVersion_1', 'test_probability'], axis=1)

        adv_sampler = self._set_adv_sampler(df_train)
        if 'CatBoost_' in self.name:
            print('FillNA for CatBoost')
            X_train = self._fillna_and_convert_float(X_train)
            X_test = self._fillna_and_convert_float(X_test)
            test_mode = 2

        # exec_lightgbm
        val_score, pred, oof = self._cv(
            X_train, df_train, X_test, adv_sampler, test_mode)
        print('### val_score: {}'.format(val_score))

        # save pickle files for next trial
        if test_mode != 1:
            self._save_trained_data(val_score, pred, oof, df_train, df_test)
        return oof, pred

    def fit_predict_for_tester(self, test_mode=False):
        df_train, df_test = self._load_data()
        val_score, pred, oof = self._cv(X_train, df_train, X_test, test_mode)
        print('### val_score: {}'.format(val_score))
        return oof, pred

    def _cv(self, X_train, df_train, X_test, adv_sampler, test_mode=False):
        """
        execute cross_validation
        self._predict: 各モデルを呼び出す
        """
        oof = np.zeros(len(X_train))
        pred = np.zeros(len(X_test))
        y_train = df_train['HasDetections']
        random.seed(42)

        for i in range(self.n_stacks):
            params = self._set_params(pattern=self.param_pattern, i=i)
            folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=15*i)
            for fold_, (train_idx, valid_idx) in \
                            enumerate(folds.split(X_train, y_train)):
                if test_mode == 2:
                    matcher = ''.join(random.sample('0123456789abcdef', 2))
                    train_sampler = df_train['MachineIdentifier'] \
                                            .str.match('.+[{}]$'.format(matcher))
                else:
                    train_sampler = None

                oof, pred = self._predict(
                    X_train, y_train, train_idx, valid_idx,
                    self.n_splits, self.n_stacks, oof, pred,
                    params, X_test, adv_sampler, train_sampler)

        return roc_auc_score(y_train[adv_sampler], oof[adv_sampler]), pred, oof

    def _set_adv_sampler(self, df_train_for_adv):
        """ AdversarialValidation対応 """
        random.seed(42)
        matcher = ''.join(random.sample('0123456789abcdef', 4))
        adv_sampler = \
            (
                (df_train_for_adv['AvSigVersion_1'] != 275) & \
                (df_train_for_adv['MachineIdentifier'].str.match('.+[{}]$'.format(matcher))) & \
                (df_train_for_adv['test_probability'] > 0.1)
            ) | \
            (df_train_for_adv['AvSigVersion_1'] == 275)
        return adv_sampler

    def _load_trained_data(self):
        oof = load_calculated_model_data(self.name, 'oof')
        pred = load_calculated_model_data(self.name, 'pred')
        return oof, pred

    def _save_trained_data(self, val_score, pred, oof, df_train, df_test):
        df_oof = pd.DataFrame({
            "machine_id": df_train['machine_id'].values
        })
        df_oof["HasDetections"] = oof
        df_oof = reduce_mem_usage(df_oof)
        df_oof.to_pickle(
            "./data/algos/{}_{}_cv_{}.pickle"\
                .format(self.name, 'oof', val_score))

        df_pred = pd.DataFrame({
            "machine_id": df_test['machine_id'].values
        })
        df_pred["HasDetections"] = pred
        df_pred = reduce_mem_usage(df_pred)
        df_pred.to_pickle(
            "./data/algos/{}_{}_cv_{}.pickle"\
                .format(self.name, 'pred', val_score))

    def _load_data(self, test_mode):
        df_train, df_test, _ = load_data_for_tester(
            scaled=self.scale_type, test_mode=test_mode)
        df_train = df_train.loc[:, sorted(df_train.columns)]
        df_test = df_test.loc[:, sorted(df_test.columns)]

        return df_train, df_test

    def _fillna_and_convert_float(self, df):
        category_cols = [col for col in df.columns
                         if re.search(r"_category$", col)]
        category_cols += list(df.select_dtypes('category').columns)
        ng_columns = ['MachineIdentifier', 'machine_id',
                      'AvSigVersion_1', 'test_probability', 'HasDetections']
        num_cols = list(set(df.columns) - set(category_cols) - set(ng_columns))

        for col in category_cols:
            if df[col].dtype.name == 'category':
                df[col].cat.add_categories(-1, inplace=True)
                df[col].fillna(-1, inplace=True)
            else:
                df[col].fillna(-1, inplace=True)

        for col in num_cols:
            if df[col].dtype.name == 'float32':
                df[col] = df[col].astype('float64')
            else:
                df[col] = df[col].astype('float32')

        unnecesary_columns = ['Census_PrimaryDiskTypeName','Census_MDC2FormFactor','SkuEdition','LDA_AVProductStatesIdentifier_by_Census_OEMModelIdentifier_3','Census_IsSecureBootEnabled','Census_OSArchitecture','LDA_AVProductStatesIdentifier_by_CountryIdentifier_2','LDA_CountryIdentifier_by_AVProductStatesIdentifier_0','Census_ProcessorCoreCount','LDA_Census_FirmwareVersionIdentifier_by_Census_OEMModelIdentifier_2','OSBuild_freq_percentile','Census_OSVersion_rounded_build_6_num','EngineVersion_rounded_build_num','OsBuildLab_ReleaseYearMonth','Census_OSEdition_categorized_category','LDA_Census_OEMModelIdentifier_by_Census_FirmwareVersionIdentifier_3','LDA_Census_FirmwareVersionIdentifier_by_Census_OEMModelIdentifier_1','ScreenRatio_TE','LDA_Census_OEMModelIdentifier_by_Census_FirmwareVersionIdentifier_0','LDA_Census_OEMModelIdentifier_by_Census_FirmwareVersionIdentifier_4','LDA_Census_FirmwareVersionIdentifier_by_Census_OEMModelIdentifier_0','Census_OSVersion_rounded_build_num_category','LDA_SmartScreen_by_AVProductStatesIdentifier_3','LDA_Census_FirmwareVersionIdentifier_by_Census_OEMModelIdentifier_4','Census_IsTouchEnabled','LDA_Census_OEMModelIdentifier_by_Census_FirmwareVersionIdentifier_1','LDA_Census_OEMModelIdentifier_by_Census_FirmwareVersionIdentifier_2','Census_InternalPrimaryDisplayResolutionVertical_num','Census_ProcessorClass','OSBranch_3_category','Census_InternalPrimaryDisplayResolutionHorizontal_num','LDA_Census_FirmwareVersionIdentifier_by_Census_OEMModelIdentifier_3','Census_PowerPlatformRoleName','CountryIdentifier_BE_HighDetectionCountries','OSBranch_2_category','AVProductsInstalled','ScreenQuality_category','IsProtected','LDA_SmartScreen_by_AVProductStatesIdentifier_2','UacLuaenable','OSBranch_head_2_category','Census_IsAlwaysOnAlwaysConnectedCapable','Census_OSVersion_rounded_build_num','OSBranch_head_category','LDA_CountryIdentifier_by_AVProductStatesIdentifier_4','YearOfAvSigversion_num','HasTpm','OSVer_unmatch_num','LDA_CountryIdentifier_by_AVProductStatesIdentifier_3','Census_IsVirtualDevice','SMode']
        df.drop(unnecesary_columns, axis=1, inplace=True)

        return df
