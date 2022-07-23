

import pandas as pd
import pandas
import scipy
import numpy
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
import collections
import numpy as np
class StressPrediction:
    def __init__(self):
        self.dataset = pd.read_csv('./Dataset/out.csv', index_col=0)
        self.dataset.drop(columns=['begin', 'end'], inplace=True)


if __name__ == '__main__':
    obj = StressPrediction()
    print(obj.dataset.info())
    # separate array into input and output components
    X = obj.dataset.drop(columns=['label'],axis=0)
    Y = obj.dataset['label']

    # initialising the StandardScaler
    scaler = StandardScaler().fit(X)
    rescaledX = scaler.transform(X)

    # summarize transformed data
    numpy.set_printoptions(precision=3)
    print(rescaledX[0:5, :])

