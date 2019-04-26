import sys
import warnings
from sklearn.linear_model import LogisticRegression

from ensemble import Ensemble
from algos.lightgbm import MyLightGBM
from algos.kernels import Kernels
#from algos.catboost import MyCatBoost
#from algos.sklearn import MySklearn
#from algos.xgboost import MyXGBoost
from utils.save_files import save_commits

warnings.filterwarnings("ignore")

if __name__ == '__main__':
    test_mode = 0  # 0: FullTest / 1: run_rest (5000) / 2: train_sampler

    base_models = [
        MyLightGBM(param_pattern=0),  # 深い / baggingあり
        MyLightGBM(param_pattern=1),  # 普通
        MyLightGBM(param_pattern=2),  # 超浅い
        Kernels('Kernel_nffm'),
        Kernels('Kernel_xdeepfm'),
        #MySklearn('RandomForest'),
        #MySklearn('ExtraTrees'),
        #MyXGBoost(),
        #MyCatBoost(),
    ]

    stacker = LogisticRegression()
    stack = Ensemble(n_splits=3, n_stacks=5,
                     stacker=stacker, base_models=base_models,
                     use_rank=True, use_adv_train=False,
                     corr_threshold=1.0, test_mode=test_mode)
    val_score, oof, pred, df_test = stack.fit_predict()

    save_commits(df_test, pred, "./commits/stack_{}.csv".format(val_score))
