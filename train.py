import numpy as np
import pickle
import h5py as h5
import os.path
from keras.models import Sequential, load_model
from keras.utils import to_categorical
from kapre.time_frequency import Spectrogram
from kapre.utils import Normalization2D
from keras.layers import Dense, Conv2D, Flatten, Dropout
from keras.layers.convolutional import MaxPooling2D
from keras import backend
backend.clear_session()

sr = 44100
input_shape = (1, sr*30)

train_size = 500
test_size = 100

with h5.File("dataset.h5") as h5f:
    data = h5f["data"]
    labels = h5f["labels"]

    X_train = np.divide(data[0:train_size, :], 2**15)\
        .reshape(train_size, 1, sr*30)
    X_test = np.divide(data[500:500+test_size, :], 2**15)\
        .reshape(test_size, 1, sr*30)

    Y_train = to_categorical(labels[0:train_size])
    Y_test = to_categorical(labels[500:500+test_size])


if os.path.isfile("dataset.h5"):
    model = load_model("dataset.h5")
    model.load_weights('dataset.h5')
else:
    model = Sequential()

    model.add(Spectrogram(n_dft=512, n_hop=None, padding='same',
                          power_spectrogram=2.0,
                          return_decibel_spectrogram=True,
                          trainable_kernel=False,
                          image_data_format='default'))
    model.add(Normalization2D(str_axis='freq'))

    model.add(Conv2D(16, kernel_size=3, activation="relu"))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(16, kernel_size=3, activation="relu"))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(16, kernel_size=3, activation="relu"))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(16, kernel_size=3, activation="relu"))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(256, activation="relu"))
    model.add(Dropout(0.2))
    model.add(Dense(256, activation="relu"))
    model.add(Dropout(0.2))
    model.add(Dense(4, activation="softmax"))

    model.compile(optimizer='adam', loss='categorical_crossentropy',
                  metrics=['categorical_accuracy'], shuffle=True)

history = model.fit(X_train, Y_train, validation_data=(X_test, Y_test),
                    batch_size=10, epochs=10)

model.save("model.h5")
numh = 1
while os.path.isfile(f"history{numh}.pickle"):
    numh += 1

with open(f'history{numh}.pickle', 'wb') as file_pi:
    pickle.dump(history.history, file_pi)

