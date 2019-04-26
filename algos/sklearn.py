import numpy as np
from .base import BaseModel

class MySklearn(BaseModel):
    def __init__(self, model, param_pattern=0, n_stacks=1):
        super().__init__(param_pattern, n_stacks)
        self.name = '{}_{}'.format(model, param_pattern)
        self.model = self._set_model(model)
        self.scale_type = True

    def _predict(self, X_train, y_train, train_idx, valid_idx,
                 splits, stacks, oof, pred, params, X_test,
                 adv_sampler, train_sampler=None):
        if train_sampler is not None:
            X_train_fold = X_train.iloc[train_idx][train_sampler]
            y_train_fold = y_train.iloc[train_idx][train_sampler]
        else:
            X_train_fold = X_train.iloc[train_idx]
            y_train_fold = y_train.iloc[train_idx]

        X_val = X_train.iloc[valid_idx][adv_sampler]
        y_val = y_train.iloc[valid_idx][adv_sampler]

        clf = self.model
        clf.fit(X_train_fold, y_train_fold)
        pred += \
            np.array([p[1] for p in clf.predict_proba(X_test)]) / splits / stacks
        oof[valid_idx] += \
            np.array([p[1] for p in clf.predict_proba(X_train.iloc[valid_idx])]) / stacks

        return oof, pred

    def _set_model(self, model):
        if model == 'RandomForest':
            from sklearn.ensemble import RandomForestClassifier
            return RandomForestClassifier(n_estimators=500,
                                          max_depth=10,
                                          min_samples_split=150,
                                          min_samples_leaf=1000,
                                          random_state=42,
                                          n_jobs=-1)
        elif model == 'ExtraTrees':
            from sklearn.ensemble import ExtraTreesClassifier
            return ExtraTreesClassifier(n_estimators=300,
                                        max_features=0.3,
                                        max_depth=8,
                                        min_samples_split=400,
                                        min_samples_leaf=500,
                                        random_state=42,
                                        n_jobs=-1)

    def _set_params(self, pattern=0, i=0):
        return None
