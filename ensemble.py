import numpy as np
import pandas as pd
import random
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import roc_auc_score
from utils.load_data import load_data_for_tester

class Ensemble:
    def __init__(self, n_splits, n_stacks, stacker, base_models,
                 use_rank=True, use_adv_train=False,
                 corr_threshold=0.95, test_mode=False):
        self.name = stacker.__class__.__name__
        self.n_splits = n_splits
        self.n_stacks = n_stacks
        self.stacker = stacker
        self.base_models = base_models
        self.use_rank = use_rank
        self.use_adv_train = use_adv_train
        self.corr_threshold = corr_threshold
        self.test_mode = test_mode

    def fit_predict(self):
        print('################### Stacking Using: {}'.format(self.name))
        df_train_for_adv, df_test_for_submission, _ = load_data_for_tester(
                scaled=False, test_mode=self.test_mode)
        df_train_for_adv = df_train_for_adv.loc[:, ['HasDetections', 'AvSigVersion_1', 'MachineIdentifier', 'test_probability'] ]
        df_test_for_submission = df_test_for_submission.loc[:, ['MachineIdentifier', 'machine_id']]

        y_train = df_train_for_adv['HasDetections']
        X_train = pd.DataFrame()
        df_test = pd.DataFrame()
        metrics = {}

        for i, model in enumerate(self.base_models):
            oof, pred = model.fit_predict(self.test_mode)
            X_train[model.name] = oof   # 1段目のoof → 2段目のtrain
            df_test[model.name] = pred  # 1段目のpred
            metrics[model.name] = self._calc_metrics(df_train_for_adv, y_train, oof)

        if self.use_rank:
            X_train = X_train.rank() / len(X_train)
            df_test = df_test.rank() / len(df_test)

        # 不要なモデルの間引き
        unnecessary_models = self._find_unnecessary_models(X_train, metrics)
        if len(unnecessary_models) > 0:
            X_train, df_test, metrics = self._eliminate_unnecessary_models(
                X_train, df_test, metrics, unnecessary_models)

        print('====== X_train.head()')
        print(X_train.head())

        print('====== df_test.head()')
        print(df_test.head())

        print('====== metrics')
        print(metrics)

        oof_stack = np.zeros(len(X_train))
        pred_stack = np.zeros(len(df_test))
        coefs = np.zeros(len(X_train.columns))
        for i in range(self.n_stacks):
            folds = StratifiedKFold(n_splits=self.n_splits, shuffle=True,
                          random_state=15*(i+1))
            for fold_, (train_idx, valid_idx) in \
                            enumerate(folds.split(X_train, y_train)):
                if self.use_adv_train:
                    matcher = ''.join(random.sample('0123456789abcdef', 4))
                    adv_sampler = \
                        (
                            (df_train_for_adv['AvSigVersion_1'] != 275) & \
                            (df_train_for_adv['MachineIdentifier'].str.match('.+[{}]$'.format(matcher))) & \
                            (df_train_for_adv['test_probability'] > 0.1)
                        ) | \
                        (df_train_for_adv['AvSigVersion_1'] == 275)
                    self.stacker.fit(
                        X_train.iloc[train_idx][adv_sampler],
                        y_train.iloc[train_idx][adv_sampler])
                else:
                    self.stacker.fit(
                        X_train.iloc[train_idx], y_train.iloc[train_idx])

                pred_stack += \
                    self._predict(df_test) / self.n_splits / self.n_stacks
                oof_stack[valid_idx] += \
                    self._predict(X_train.iloc[valid_idx]) / self.n_stacks
                coefs += self.stacker.coef_[0] / self.n_splits / self.n_stacks

        print('====== pred_stack[:100]')
        print(pred_stack[:100])
        print('====== coefs_')
        print(['{}: {}'.format(model, round(coef, 3))
               for coef, model in zip(coefs, X_train.columns)])
        print('====== model_summary')
        print('use_model: {}'.format(X_train.columns))
        val_score = self._calc_metrics(df_train_for_adv, y_train, oof_stack)
        print('CV: {}'.format(val_score))

        return val_score, oof_stack, pred_stack, df_test_for_submission

    def _predict(self, df):
        return np.array([p[1] for p in self.stacker.predict_proba(df)])

    def _calc_metrics(self, df_train_for_adv, y_train, oof):
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
        return roc_auc_score(y_train[adv_sampler], oof[adv_sampler])

    def _find_unnecessary_models(self, X_train, metrics):
        unnecessary_models = set()
        df_corr = X_train.corr()
        print(df_corr)

        upper = df_corr.where(np.triu(np.ones(df_corr.shape), k=1).astype(np.bool))
        for column in upper:
            for column_b in upper[column][upper[column] > self.corr_threshold].index:
                # 他の課題でやるときは注意
                if metrics[column_b] < metrics[column]:
                    unnecessary_models.add(column_b)
                else:
                    unnecessary_models.add(column)

        if len(unnecessary_models) > 0:
            print('unnecessary_models: {}'.format(unnecessary_models))
        return unnecessary_models

    def _eliminate_unnecessary_models(self, X_train, df_test,
                                      metrics, unnecessary_models):
        X_train.drop(unnecessary_models, axis=1, inplace=True)
        df_test.drop(unnecessary_models, axis=1, inplace=True)
        for k in unnecessary_models:
            metrics.pop(k, None)

        return X_train, df_test, metrics
