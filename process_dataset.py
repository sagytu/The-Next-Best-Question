import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from collections import defaultdict

pd.options.mode.chained_assignment = None


def read_preprocess(dataset):
    if dataset.lower() == 'mnist':
        return read_preprocess_mnist()
    elif dataset.lower() == 'miniboone':
        return read_preprocess_miniboone()
    elif dataset.lower() == 'parkinson':
        return read_preprocess_parkinson()
    elif dataset.lower() == 'cardio':
        return read_preprocess_cardio()
    elif dataset.lower() == 'statlog':
        return read_preprocess_statlog()
    elif dataset.lower() == 'spambase':
        return read_preprocess_spambase()


def read_preprocess_cardio():
    cardio = pd.read_excel('datasets\\cardiotocography\\CTF_data.xlsx')

    to_use_feats = ['LB', 'AC', 'FM', 'UC', 'DL', 'DS', 'DP', 'ASTV', 'MSTV', 'ALTV', 'MLTV', 'Width', 'Min',
                    'Max', 'Nmax', 'Nzeros', 'Mode', 'Mean', 'Median', 'Variance', 'Tendency']

    X = cardio[to_use_feats]
    y = cardio['CLASS']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.35, random_state=10, stratify=y)

    scalers_d = defaultdict(MinMaxScaler)
    to_scale = list(X.columns)

    for c in to_scale:
        X_train[c] = scalers_d[c].fit_transform(X_train[c].values[:, np.newaxis])
        X_test[c] = scalers_d[c].transform(X_test[c].values[:, np.newaxis])

    # Set the columns names as numbers to be used for indexing
    X_train.columns = list(range(X_train.shape[1]))
    X_test.columns = list(range(X_test.shape[1]))

    X_train.reset_index(drop=True, inplace=True)
    X_test.reset_index(drop=True, inplace=True)
    y_train.reset_index(drop=True, inplace=True)
    y_test.reset_index(drop=True, inplace=True)

    return X_train, X_test, y_train, y_test


def read_preprocess_parkinson():

    df_parkinson = pd.read_csv('datasets\\parkinson\\pd_speech_features_parkinson.csv')
    header = df_parkinson.iloc[0]
    df_parkinson = df_parkinson[1:]
    df_parkinson.columns = header

    df_parkinson.drop('id', axis=1, inplace=True)
    df_parkinson.drop('gender', axis=1, inplace=True)

    for c in df_parkinson.columns:
        df_parkinson[c] = df_parkinson[c].astype(float)

    X = df_parkinson[[x for x in df_parkinson.columns if x != 'class']]
    y = df_parkinson['class']

    # split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=10, stratify=y)

    scalers_d = defaultdict(MinMaxScaler)
    to_scale = list(df_parkinson.columns)
    to_scale.remove('class')

    for c in to_scale:
        X_train[c] = scalers_d[c].fit_transform(X_train[c].values[:, np.newaxis])
        X_test[c] = scalers_d[c].transform(X_test[c].values[:, np.newaxis])

    # Set the columns names as numbers to be used for indexing
    X_train.columns = list(range(X_train.shape[1]))
    X_test.columns = list(range(X_test.shape[1]))

    X_train.reset_index(drop=True, inplace=True)
    X_test.reset_index(drop=True, inplace=True)
    y_train.reset_index(drop=True, inplace=True)
    y_test.reset_index(drop=True, inplace=True)

    return X_train, X_test, y_train, y_test


def read_preprocess_miniboone():
    # The code used in "Classification with Costly Features using Deep Reinforcement Learning"
    COL_LABEL = '_label'

    SEED = 998822
    # ---
    np.random.seed(SEED)

    data = pd.read_csv('datasets\\miniboone\\MiniBooNE_PID.txt', header=None, sep=' +')

    data[COL_LABEL] = 0
    data.iloc[36500:][COL_LABEL] = 1

    data = data[data[0] > -900]

    data.iloc[:, 0:-1] = data.iloc[:, 0:-1].astype('float32')
    data.iloc[:, -1:] = data.iloc[:, -1:].astype('int32')


    TRAIN_SIZE = 45359
    VAL_SIZE = 19439
    TEST_SIZE = 64798

    # JUST FOR RUNNING FAST FOR TESTING THE PIPELINE
    # TRAIN_SIZE = 1000
    # VAL_SIZE = 2000
    # TEST_SIZE = 3000

    data = data.sample(frac=1)

    data_train = data.iloc[0:TRAIN_SIZE]
    data_val = data.iloc[TRAIN_SIZE:TRAIN_SIZE + VAL_SIZE]
    data_test = data.iloc[TRAIN_SIZE + VAL_SIZE:]

    # My addition of scaling for the KNN algorithm
    scalers_d = defaultdict(MinMaxScaler)
    to_scale = list(data_train.columns)
    to_scale.remove('_label')

    for c in to_scale:
        data_train[c] = scalers_d[c].fit_transform(data_train[c].values[:, np.newaxis])
        data_test[c] = scalers_d[c].transform(data_test[c].values[:, np.newaxis])

    X_train, y_train = data_train[[i for i in range(50)]], data_train['_label']
    X_test, y_test = data_test[[i for i in range(50)]], data_test['_label']

    # Set the columns names as numbers to be used for indexing
    X_train.columns = list(range(X_train.shape[1]))
    X_test.columns = list(range(X_test.shape[1]))

    # split data
    X_train.reset_index(drop=True, inplace=True)
    X_test.reset_index(drop=True, inplace=True)
    y_train.reset_index(drop=True, inplace=True)
    y_test.reset_index(drop=True, inplace=True)

    X_train = X_train
    X_test = X_test
    y_train = y_train
    y_test = y_test
    return X_train, X_test, y_train, y_test


def read_preprocess_mnist():
    data_train = pd.read_csv("datasets\\mnist\\mnist-train.csv", sep=',', index_col=0)
    data_test = pd.read_csv("datasets\\mnist\\mnist-test.csv", sep=',', index_col=0)

    x_train, y_train = data_train.iloc[:, :784], data_train['_label']
    x_test, y_test = data_test.iloc[:, :784], data_test['_label']

    # My addition of scaling for the KNN algorithm
    scalers_d = defaultdict(MinMaxScaler)
    to_scale = list(x_train.columns)

    for c in to_scale:
        x_train[c] = scalers_d[c].fit_transform(x_train[c].values[:, np.newaxis])
        x_test[c] = scalers_d[c].transform(x_test[c].values[:, np.newaxis])

    # Set the columns names as numbers to be used for indexing
    x_train.columns = list(range(x_train.shape[1]))
    x_test.columns = list(range(x_test.shape[1]))

    x_train.reset_index(drop=True, inplace=True)
    x_test.reset_index(drop=True, inplace=True)
    y_train.reset_index(drop=True, inplace=True)
    y_test.reset_index(drop=True, inplace=True)

    # Take Sample for test
    x_train = x_train
    x_test = x_test
    y_train = y_train
    y_test =y_test

    return x_train, x_test, y_train, y_test


def read_preprocess_statlog():
    statlog = pd.read_csv('datasets\\statlog\\heart.dat', header=None, sep=' ')
    X = statlog[list(range(13))]
    y = statlog[13]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=10, stratify=y)

    scalers_d = defaultdict(MinMaxScaler)
    to_scale = list(statlog.columns)
    to_scale.remove(13)

    for c in to_scale:
        X_train[c] = scalers_d[c].fit_transform(X_train[c].values[:, np.newaxis])
        X_test[c] = scalers_d[c].transform(X_test[c].values[:, np.newaxis])

    # Set the columns names as numbers to be used for indexing
    X_train.columns = list(range(X_train.shape[1]))
    X_test.columns = list(range(X_test.shape[1]))

    X_train.reset_index(drop=True, inplace=True)
    X_test.reset_index(drop=True, inplace=True)
    y_train.reset_index(drop=True, inplace=True)
    y_test.reset_index(drop=True, inplace=True)

    return X_train, X_test, y_train, y_test


def read_preprocess_spambase():
    spambase = pd.read_csv('datasets\\Spambase\\spambase.data', header=None)
    X = spambase[list(range(57))]
    y = spambase[57]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=10, stratify=y)

    scalers_d = defaultdict(MinMaxScaler)
    to_scale = list(spambase.columns)
    to_scale.remove(57)

    for c in to_scale:
        X_train[c] = scalers_d[c].fit_transform(X_train[c].values[:, np.newaxis])
        X_test[c] = scalers_d[c].transform(X_test[c].values[:, np.newaxis])

    # Set the columns names as numbers to be used for indexing
    X_train.columns = list(range(X_train.shape[1]))
    X_test.columns = list(range(X_test.shape[1]))

    X_train.reset_index(drop=True, inplace=True)
    X_test.reset_index(drop=True, inplace=True)
    y_train.reset_index(drop=True, inplace=True)
    y_test.reset_index(drop=True, inplace=True)

    return X_train, X_test, y_train, y_test









