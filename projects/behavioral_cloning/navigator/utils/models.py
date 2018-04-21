from keras import regularizers
from keras.layers import Conv2D, Dense, Dropout, Flatten, Lambda
from keras.models import Sequential
from keras.optimizers import Adam


def get_nvidia_model(dropout_prob=None, learning_rate=0.001, regularize_val=0.001):

    input_shape = (66, 200, 3)
    model = Sequential()

    # Normalization layer to make use of GPU acceleration
    model.add(Lambda(lambda x: x / 127.5 - 1.0, input_shape=input_shape))

    # starts with five convolutional and maxpooling layers
    model.add(Conv2D(24, (5, 5), padding='valid', strides=(2, 2),
                     activation='relu'))  # , kernel_regularizer=regularizers.l2(0.01)))
    model.add(Conv2D(36, (5, 5), padding='valid', strides=(2, 2),
                     activation='relu'))  #, kernel_regularizer=regularizers.l2(0.01)))
    model.add(Conv2D(48, (5, 5), padding='valid', strides=(2, 2),
                     activation='relu'))  #, kernel_regularizer=regularizers.l2(0.01)))
    model.add(Conv2D(64, (3, 3), padding='valid', strides=(1, 1),
                     activation='relu'))  #, kernel_regularizer=regularizers.l2(0.01)))
    model.add(Conv2D(64, (3, 3), padding='valid', strides=(1, 1),
                     activation='relu'))  #, kernel_regularizer=regularizers.l2(0.01)))
    if dropout_prob:
        model.add(Dropout(dropout_prob))  # Fraction of the input units to drop

    model.add(Flatten())
    # Next, five fully connected layers
    model.add(Dense(100, activation='relu')) #, kernel_regularizer=regularizers.l2(0.01)))
    model.add(Dense(50, activation='relu',)) # kernel_regularizer=regularizers.l2(0.01)))
    model.add(Dense(10, activation='relu',)) # kernel_regularizer=regularizers.l2(0.01)))
    model.add(Dense(1))
    model.summary()

    model.compile(loss='mse', optimizer=Adam(lr=learning_rate))

    return model, input_shape
