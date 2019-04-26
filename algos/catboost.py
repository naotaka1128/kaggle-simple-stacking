import re
import catboost
import numpy as np
from .base import BaseModel

class MyCatBoost(BaseModel):
    def __init__(self, param_pattern=0):
        super().__init__(param_pattern)
        self.name = 'CatBoost_{}'.format(param_pattern)
        self.scale_type = False

    def _predict(self, X_train, y_train, train_idx, valid_idx, splits, stacks,
                 oof, pred, params, X_test, adv_sampler, train_sampler=None):
        category_cols = [col for col in X_train.columns
                         if re.search(r"_category$", col)]
        category_cols += list(X_train.select_dtypes('category').columns)
        categorical_features_indices = [
            list(X_train.columns).index(i) for i in category_cols]

        if train_sampler is not None:
            X_train_fold = X_train.iloc[train_idx][train_sampler]
            y_train_fold = y_train.iloc[train_idx][train_sampler]
        else:
            X_train_fold = X_train.iloc[train_idx]
            y_train_fold = y_train.iloc[train_idx]

        X_val = X_train.iloc[valid_idx][adv_sampler]
        y_val = y_train.iloc[valid_idx][adv_sampler]

        print('      Start CatBoostClassifier')
        clf = catboost.CatBoostClassifier(
            use_best_model=True,
            iterations=500,
            learning_rate=0.1,
            loss_function='Logloss',
            eval_metric='AUC',
            verbose=True,
            random_seed=42
        )
        clf.fit(X_train_fold, y_train_fold,
                eval_set=(X_val, y_val),
                cat_features=categorical_features_indices,
                early_stopping_rounds=20)
        pred += \
            np.array([p[1] for p in clf.predict_proba(X_test)]) / splits / stacks
        oof[valid_idx] += \
            np.array([p[1] for p in clf.predict_proba(X_train.iloc[valid_idx])]) / stacks

        return oof, pred

    def _set_params(self, pattern=0, i=0):
        return None
