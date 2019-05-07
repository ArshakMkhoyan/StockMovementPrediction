import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler



def movement_close(df):
    difference=df.Close.values[:-1]-df.Close.values[1:]
    difference=difference.reshape((1,-1))
    target=difference<0
    df_target=pd.DataFrame({'Target':target.flatten()}, index=df.index[:-1])
    return df.join(df_target)

def split_target(df):
    y, x=df['Target'], df.drop(['Target'], axis=1)
    return (y, x)
        
def split_data(y, x, ratio=0.85, batch_size=30):
    to_train = int(len(x) * ratio)
    to_train -= to_train % batch_size

    x_train = x[:to_train]
    y_train = y[:to_train]
    x_test = x[to_train:]
    y_test = y[to_train:]

    # tweak to match with batch_size
    to_drop = x_test.shape[0] % batch_size
    if to_drop:
        x_test = x_test[:-1 * to_drop]
        y_test = y_test[:-1 * to_drop]
    
    return (x_train, y_train), (x_test, y_test)


def scale_data(x_train, x_test):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_train = scaler.fit_transform(x_train)
    scaled_test=scaler.transform(x_test)
    return (scaled_train, scaled_test)

def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = pd.DataFrame(data)
    cols, names = list(), list()
    
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
    agg = pd.concat(cols, axis=1)
    agg.columns = names
    
    if dropnan:
        agg.dropna(inplace=True)
        
    return agg

def reshape_data(scaled_train, scaled_test, y_train, y_test, n_days=5):
    # specify the number of lag hours
    n_days = 5   
    n_features = len(scaled_train[0])
    # frame as supervised learning
    reframed_train_x = series_to_supervised(scaled_train, n_days, n_features).values
    reframed_test_x = series_to_supervised(scaled_test, n_days, n_features).values
    reframed_train_x = reframed_train_x.reshape((reframed_train_x.shape[0], n_days, n_features))
    reframed_test_x = reframed_test_x.reshape((reframed_test_x.shape[0],  n_days, n_features))
    
    reframed_train_y=y_train.shift(1)[n_days:].values
    reframed_test_y=y_test.shift(1)[n_days:].values

    print(f'Train X shape: {reframed_train_x.shape}\nTrain Y shape: {reframed_train_y.shape} \nTest X shape:{reframed_test_x.shape} \nTest Y shape: {reframed_test_y.shape}')
    
    return (reframed_train_x, reframed_train_y), (reframed_test_x, reframed_test_y)

def ready_data(df, split_ratio=0.85, batch_size=30, n_days=5):
    
    df=movement_close(df)
    y, x = split_target(df)    
    (x_train, y_train), (x_test, y_test) = split_data(y, x, ratio=split_ratio, batch_size=batch_size)
    scaled_train, scaled_test = scale_data(x_train, x_test)
    (reframed_train_x, reframed_train_y), (reframed_test_x, reframed_test_y) = reshape_data(scaled_train, scaled_test, y_train, y_test, n_days=n_days)
    print(f'\nProportion of Target variable. \nTest set: {np.round(np.mean(y_test), decimals=4)} \nTrain set: {np.round(np.mean(y_train), decimals=4)}\n')
    
    return (reframed_train_x, reframed_train_y), (reframed_test_x, reframed_test_y)