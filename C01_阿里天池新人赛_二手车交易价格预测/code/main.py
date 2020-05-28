import numpy as np
import pandas as pd
from C01_阿里天池新人赛_二手车交易价格预测.model import gbdt_model


def load_predict_data():
    data_frame = pd.read_csv('../data/used_car_testB_20200421.csv')
    src_data = data_frame.values
    np_data = np.empty(shape=(50000, 30), dtype=float)
    for row_index in range(src_data.shape[0]):
        words_list = src_data[row_index, 0].split(sep=' ')
        if words_list.__contains__('-') or words_list.__contains__(''):
            for index in range(len(words_list)):
                e = words_list[index]
                if e.__eq__('-'):
                    words_list[index] = '-1'
                elif e.__eq__(''):
                    words_list[index] = '-2'
        np_data[row_index] = np.array(words_list).reshape(1, -1)

    col_list = [3, 5, 6, 7]
    delete_row = []
    for row in range(np_data.shape[0]):
        for col in col_list:
            if np_data[row, col] - (-2) < 1e-8:
                delete_row.append(row)
        # index=8 大于600则当做600处理
        if np_data[row, 8] > 600:
            np_data[row, 8] = 600
    # index=10 不填默认为有损坏, 所有-1变为0
    data_col_10 = np_data[:, 10]
    data_col_10[data_col_10 < 0] = 0
    np_data[:, 10] = data_col_10

    return np_data[:, 0], np_data[:, 1:]


if __name__ == '__main__':
    model = gbdt_model.get_model()
    print(model)

    X_index, X = load_predict_data()
    print(X.shape)

    y_predict = model.predict(X)
    print(y_predict.shape)

    pandas_data = pd.read_csv('../data/used_car_sample_submit.csv')
    print(pandas_data.shape)

    pandas_data['SaleID'] = pandas_data['SaleID'] + 50000
    pandas_data['price'] = y_predict
    pandas_data['price'] = pandas_data['price'].astype('int')
    pandas_data.to_csv('../data/used_car_sample_submit.csv', index=False)
