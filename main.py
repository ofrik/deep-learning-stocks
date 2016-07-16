from pandas_datareader.data import DataReader
from datetime import datetime
import os
import pandas as pd

import random
import numpy as np
from keras.models import Sequential
from keras.layers.recurrent import LSTM
from keras.layers.core import Dense, Activation, Dropout
from sklearn.cross_validation import train_test_split

MAX_WINDOW = 10


def get_data_if_not_exists():
    if os.path.exists("./data/ibm.csv"):
        return pd.read_csv("./data/ibm.csv")
    else:
        if not os.path.exists("./data"):
            os.mkdir("data")
        ibm_data = DataReader('IBM', 'yahoo', datetime(1950, 1, 1), datetime.today())
        pd.DataFrame(ibm_data).to_csv("./data/ibm.csv")
        return pd.DataFrame(ibm_data)


def extract_features(items):
    return [[0, 1] if item[4] > item[6] else [1, 0] for item in items]


def extract_expected_result(item):
    return [0, 1] if item[4] > item[6] else [1, 0]


def generate_input_and_outputs(data):
    step = 1
    inputs = []
    outputs = []
    for i in range(0, len(data) - MAX_WINDOW, step):
        inputs.append(extract_features(data.iloc[i:i + MAX_WINDOW].as_matrix()))
        outputs.append(extract_expected_result(data.iloc[i + MAX_WINDOW].as_matrix()))
    return inputs, outputs


if __name__ == "__main__":
    print "loading the data"
    data = get_data_if_not_exists()
    print "done loading the data"
    print "data columns names: %s"%data.columns.values
    print "generating model input and outputs"
    X, y = generate_input_and_outputs(data)
    print "done generating input and outputs"
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    layer_output_size1 = 512
    layer_output_size2 = 512
    number_of_features = 5
    output_classes = len(y[0])
    percentage_of_neurons_to_ignore = 0.2
    model = Sequential()
    model.add(LSTM(layer_output_size1, return_sequences=True, input_shape=(MAX_WINDOW, len(y[0]))))
    model.add(Dropout(percentage_of_neurons_to_ignore))
    model.add(LSTM(layer_output_size2, return_sequences=False))
    model.add(Dropout(percentage_of_neurons_to_ignore))
    model.add(Dense(output_classes))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy',metrics=['accuracy'], optimizer='rmsprop')

    model.fit(X_train, y_train, batch_size=128, nb_epoch=10,validation_split=0.2)


