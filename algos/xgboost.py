import re
from xgboost import XGBClassifier
from .base import BaseModel

class MyXGBoost(BaseModel):
    def __init__(self, param_pattern=0):
        super().__init__(param_pattern)
        self.name = 'XGBoost_{}'.format(param_pattern)
        self.model = self._set_model(param_pattern)
        self.scale_type = True

    def _predict(self, X_train, y_train, train_idx, valid_idx, splits, stacks,
                 oof, pred, params, X_test, adv_sampler, train_sampler=None):
        if train_sampler is not None:
            X_train_fold = X_train.iloc[train_idx][train_sampler]
            y_train_fold = y_train.iloc[train_idx][train_sampler]
        else:
            X_train_fold = X_train.iloc[train_idx]
            y_train_fold = y_train.iloc[train_idx]

        X_val = X_train.iloc[valid_idx][adv_sampler]
        y_val = y_train.iloc[valid_idx][adv_sampler]

        clf = self.model
        clf.fit(
            X_train_fold, y_train_fold,
            eval_set=[(X_val, y_val)],
            verbose=1000, eval_metric='auc',
            early_stopping_rounds=50
        )
        pred += clf.predict(
            X_test, ntree_limit=clf.best_ntree_limit) / splits / stacks
        oof[valid_idx] += clf.predict(
            X_train.iloc[valid_idx], ntree_limit=clf.best_ntree_limit) / stacks

        return oof, pred

    def _set_model(self, pattern=0):
        if pattern == 0:
            params = {
                'silent': 1,
                'objective': 'binary:logistic',
                'eval_metric': 'auc',
                'n_estimators': 3000,
                'n_jobs': -1,
                'booster': 'gbtree',
                'alpha': 6.870660708606968e-08,
                'colsample_bytree': 0.10736688298175229,
                'eta': 0.002824400211087471,
                'gamma': 1.429198306798688e-07,
                'grow_policy': 'depthwise',
                'lambda': 12.243627242400416,
                'max_depth': 6,
                'subsample': 0.9278878775092663
            }
            return XGBClassifier(**params)
        else:
            return None

    def _set_params(self, pattern=0, i=0):
        return None
