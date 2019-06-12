from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv1D,Flatten
from keras.layers import LSTM
from keras.callbacks import EarlyStopping
from keras import regularizers

def LSTM_model(reframed_train_x, reframed_train_y, reframed_test_x, reframed_test_y, n_days, batch_size, epochs):
    look_back=n_days
    es = EarlyStopping(monitor='val_acc', baseline=0.54 ,patience=90, restore_best_weights=True)

    model = Sequential()
    model.add(LSTM(13, activation='tanh')) #, kernel_regularizer=regularizers.l1(0.0001), 13
    model.add(Dropout(0.1))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    history=model.fit(reframed_train_x, reframed_train_y, epochs=epochs, batch_size=batch_size, verbose=1, validation_data=(reframed_test_x, reframed_test_y), shuffle=False, callbacks=[es])

    return history, model