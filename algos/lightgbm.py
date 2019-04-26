import re
import lightgbm as lgb
from .base import BaseModel

class MyLightGBM(BaseModel):
    def __init__(self, param_pattern=0):
        super().__init__(param_pattern)
        self.name = 'LightGBM_{}'.format(param_pattern)
        self.scale_type = False

    def _predict(self, X_train, y_train, train_idx, valid_idx, splits, stacks,
                 oof, pred, params, X_test, adv_sampler, train_sampler=None):
        if train_sampler is not None:
            train_data = lgb.Dataset(
                X_train.iloc[train_idx][train_sampler],
                label=y_train.iloc[train_idx][train_sampler])
        else:
            train_data = lgb.Dataset(
                X_train.iloc[train_idx],
                label=y_train.iloc[train_idx])

        valid_data = lgb.Dataset(
            X_train.iloc[valid_idx][adv_sampler],
            label=y_train.iloc[valid_idx][adv_sampler])
        clf = lgb.train(params, train_data, num_boost_round=5000,
                        valid_sets = [train_data, valid_data],
                        verbose_eval=100, early_stopping_rounds=50)
        pred += clf.predict(
            X_test, num_iteration=clf.best_iteration) / splits / stacks
        oof[valid_idx] += clf.predict(
            X_train.iloc[valid_idx], num_iteration=clf.best_iteration) / stacks

        return oof, pred

    def _set_params(self, pattern=0, i=0):
        if pattern == 0:
            return {'objective': 'binary',
                    'metric': 'auc',
                    'verbosity': -1,
                    'learning_rate': 0.05,
                    'boosting': 'gbdt',
                    'feature_fraction': 0.19014769855525646,
                    'bagging_freq': 6,
                    'bagging_fraction': 0.9118200214912328,
                    'lambda_l1': 50.3006568620301,
                    'lambda_l2': 41.92315316962046,
                    'lambda_l1': 71.48698103882137,
                    'lambda_l2': 25.586471755120172,
                    'min_child_weight': 1.1993931929686235,
                    'min_data_in_leaf': 3191,
                    'num_leaves': 1926}

        elif pattern == 1:
            return {'objective': 'binary',
                    'metric': 'auc',
                    'verbosity': -1,
                    'learning_rate': 0.05,
                    'boosting': 'gbdt',
                    'feature_fraction': 0.16264967196495211,
                    'lambda_l1': 83.70403760449734,
                    'lambda_l2': 56.10523711009329,
                    'min_child_weight': 63.048351844725204,
                    'min_data_in_leaf': 1873,
                    'num_leaves': 1436}

        elif pattern == 2:
            return {'objective': 'binary',
                    'metric': 'auc',
                    'verbosity': -1,
                    'learning_rate': 0.1,
                    'boosting': 'gbdt',
                    'max_depth': 5,
                    'feature_fraction': 0.10253328121519681,
                    'bagging_freq': 3,
                    'bagging_fraction': 0.9934509167308513,
                    'lambda_l1': 0.02023708904541934,
                    'lambda_l2': 3.739597303363229,
                    'min_child_weight': 28.873876450157002,
                    'min_data_in_leaf': 3939,
                    'num_leaves': 1972}
