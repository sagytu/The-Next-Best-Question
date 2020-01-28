# Run the following script to download the MNIST dataset
from torchvision import datasets, transforms
import pandas as pd
import numpy as np

COLUMN_LABEL = '_label'
SEED = 998822
#---
np.random.seed(SEED)

#---
def get_data(train):
    data_raw = datasets.MNIST('..\\mnist\\', train=train, download=True,
                              transform=transforms.Compose([transforms.ToTensor(), lambda x: x.numpy().flatten()]),
                              target_transform=lambda y: [np.array(y)])

    data_x, data_y = zip(*data_raw)
    data_x = np.array(data_x)
    data_y = np.array(data_y, dtype='int32').reshape(-1, 1)

    data = pd.DataFrame(data_x)
    data[COLUMN_LABEL] = data_y

    return data, data_x.mean(), data_x.std()

data_train, avg, std = get_data(train=True)
data_test, _, _  = get_data(train=False)

#shuffle
val_idx = np.random.choice(data_train.shape[0], 10000, replace=False).tolist()

data_val   = data_train.iloc[val_idx]
data_train = data_train.drop(val_idx)

data_train.to_csv("..\\mnist\\mnist-train.csv")
data_val.to_csv("..\\mnist\\mnist-val.csv")
data_test.to_csv("..\\mnist\\mnist-test.csv")
