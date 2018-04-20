from keras.layers import Conv2D, Dense, Dropout, ELU, Flatten, Lambda, MaxPooling2D
from keras.models import Sequential


def get_model():

    # Our model is based on NVIDIA's "End to End Learning for Self-Driving Cars" paper
    # Source:  https://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf

    input_shape = (66, 200, 3)
    model = Sequential()

    model.add(Lambda(lambda x: x / 127.5 - 1.0, input_shape=input_shape))

    # starts with five convolutional and maxpooling layers
    model.add(Conv2D(24, (5, 5), padding='same', strides=(2, 2), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))

    model.add(Conv2D(36, (5, 5), padding='same', strides=(2, 2), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))

    model.add(Conv2D(48, (5, 5), padding='same', strides=(2, 2), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))

    model.add(Conv2D(64, (3, 3), padding='same', strides=(1, 1), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))

    model.add(Conv2D(64, (3, 3), padding='same', strides=(1, 1), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))

    model.add(Flatten())

    # Next, five fully connected layers
    model.add(Dense(1164, activation='relu'))

    model.add(Dense(100, activation='relu'))

    model.add(Dense(50, activation='relu'))

    model.add(Dense(10, activation='relu'))

    model.add(Dense(1))

    model.summary()

    model.compile(optimizer='adam', loss="mse", )

    return model, input_shape
