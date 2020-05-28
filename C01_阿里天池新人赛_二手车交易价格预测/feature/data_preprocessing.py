import pandas
import numpy as np
from collections import Counter

"""
    0 SaleID	交易ID，唯一编码
    1  name	汽车交易名称，已脱敏
    2  regDate	汽车注册日期，例如20160101，2016年01月01日
    3  model	车型编码，已脱敏
    4  brand	汽车品牌，已脱敏
    5  bodyType	车身类型：豪华轿车：0，微型车：1，厢型车：2，大巴车：3，敞篷车：4，双门汽车：5，商务车：6，搅拌车：7
    6  fuelType	燃油类型：汽油：0，柴油：1，液化石油气：2，天然气：3，混合动力：4，其他：5，电动：6
    7  gearbox	变速箱：手动：0，自动：1
    8  power	发动机功率：范围 [ 0, 600 ]
    9  kilometer	汽车已行驶公里，单位万km
    10  notRepairedDamage	汽车有尚未修复的损坏：是：0，否：1
    11  regionCode	地区编码，已脱敏
    12  seller	销售方：个体：0，非个体：1
    13  offerType	报价类型：提供：0，请求：1
    14  creatDate	汽车上线时间，即开始售卖时间
    15  price	二手车交易价格（预测目标）
    16-30  v系列特征	匿名特征，包含v0-14在内15个匿名特征
"""

titles = ['SaleID', 'name', 'regDate', 'model', 'brand', 'bodyType', 'fuelType', 'gearbox', 'power', 'kilometer', 'notRepairedDamage', 'regionCode', 'seller', 'offerType',
          'creatDate', 'price', 'v_0', 'v_1', 'v_2', 'v_3', 'v_4', 'v_5', 'v_6', 'v_7', 'v_8', 'v_9', 'v_10', 'v_11', 'v_12', 'v_13', 'v_14']


def data_transform():
    print('------------------ data transform ----------------------')
    data_frame = pandas.read_csv('../data/used_car_train_20200313.csv')
    src_data = data_frame.values
    np_data = np.empty(shape=(150000, 31), dtype=float)
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

    pandas_data = pandas.DataFrame(np_data, columns=titles)
    pandas_data.to_csv('../user_data/train_data_transform.csv', index=False)
    print('data.shape={}'.format(np_data.shape))
    print('--------------------------------------------------------')


def data_preprocessing():
    print('---------------- data preprocessing --------------------')
    data_frame = pandas.read_csv('../user_data/train_data_transform.csv')
    data = data_frame.values

    # index=3, 5, 6, 7 不填则删除
    col_list = [3, 5, 6, 7]
    delete_row = []
    for row in range(data.shape[0]):
        for col in col_list:
            if data[row, col] - (-2) < 1e-8:
                delete_row.append(row)
        # index=8 大于600则当做600处理
        if data[row, 8] > 600:
            data[row, 8] = 600

    data = np.delete(arr=data, obj=list(set(delete_row)), axis=0)

    # index=10 不填默认为有损坏, 所有-1变为0
    data_col_10 = data[:, 10]
    # counter = Counter(X_col_10)
    # print(counter)  # Counter({0.0: 111361, -1.0: 24324, 1.0: 14315})
    data_col_10[data_col_10 < 0] = 0
    data[:, 10] = data_col_10

    pandas_data = pandas.DataFrame(data, columns=titles)
    pandas_data.to_csv('../user_data/train_data_preprocessing.csv', index=False)

    print('data.shape={}'.format(data.shape))
    print('--------------------------------------------------------')


def load_data():
    print('-------------------- load data -------------------------')
    data_frame = pandas.read_csv('../user_data/train_data_preprocessing.csv')
    data = data_frame.values

    X = np.hstack([data[:, 0:15], data[:, 16:]])
    y = data[:, 15]

    print('data.shape={}'.format(data.shape))
    print('X.shape={}, y.shape={}'.format(X.shape, y.shape))
    print('--------------------------------------------------------')

    return X, y


if __name__ == '__main__':
    # data_transform()
    # data_preprocessing()
    X, y = load_data()
