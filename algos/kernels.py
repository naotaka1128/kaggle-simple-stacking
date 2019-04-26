import numpy as np
import pandas as pd
from .base import BaseModel

class Kernels(BaseModel):
    """" 学習済みの oof/pred を読み込むためだけに使う """
    def __init__(self, model, param_pattern=0):
        super().__init__(param_pattern)
        self.name = model
        self.scale_type = False

    def _predict(self, **kwargs):
        return None

    def _set_params(self, pattern=0, i=0):
        return None
