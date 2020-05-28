import numpy as np
from C01_阿里天池新人赛_二手车交易价格预测.feature import data_preprocessing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor


def get_model():
    X, y = data_preprocessing.load_data()
    X = np.delete(X, [0], axis=1)

    gb_reg = GradientBoostingRegressor(
        learning_rate=1.0,
        n_estimators=100,
        max_depth=3,
        max_leaf_nodes=None,
        min_samples_split=100,
        min_samples_leaf=1
    )
    gb_reg.fit(X, y)

    return gb_reg
