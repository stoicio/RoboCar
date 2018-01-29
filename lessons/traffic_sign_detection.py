# Load pickled data
import pickle

from keras.layers.core import Activation, Dense, Flatten
from keras.layers import Conv2D, Dropout, MaxPooling2D
from keras.models import Sequential
import numpy as np
from sklearn.preprocessing import LabelBinarizer
import tensorflow as tf


# tf.python.control_flow_ops = tf

with open('toy_dataset/small_train_traffic.p', mode='rb') as f:
    data = pickle.load(f)

X_train, y_train = data['features'], data['labels']


model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=(32, 32, 3)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.5))
model.add(Activation('relu'))
model.add(Flatten())
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dense(5))
model.add(Activation('softmax'))


# preprocess data
X_normalized = np.array(X_train / 255.0 - 0.5)


label_binarizer = LabelBinarizer()
y_one_hot = label_binarizer.fit_transform(y_train)

model.compile('adam', 'categorical_crossentropy', ['accuracy'])
history = model.fit(X_normalized, y_one_hot, nb_epoch=10, validation_split=0.2)

with open('toy_dataset/small_test_traffic.p', mode='rb') as f:
    data_test = pickle.load(f)

X_test = data_test['features']
y_test = data_test['labels']
X_normalized_test = np.array(X_test / 255 - 0.5)
y_one_hot_test = label_binarizer.fit_transform(y_test)


metrics = model.evaluate(X_normalized_test, y_one_hot_test)
for metric in range(len(model.metrics_names)):
    metric_name = model.metrics_names[metric]
    metric_value = metrics[metric]
    print('{}: {}'.format(metric_name, metric_value))
