import pickle
import time

from alexnet import AlexNet

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

import tensorflow as tf

nb_classes = 43
epochs = 10
batch_size = 64

# Load traffic signs data.
with open('./train.p', 'rb') as f:
    data = pickle.load(f)


# Split data into training and validation sets.
X_train, X_val, y_train, y_val = train_test_split(data['features'], data['labels'],
                                                  test_size=0.33, random_state=0)
# Define placeholders and resize operation.
features = tf.placeholder(tf.float32, (None, 32, 32, 3))
labels = tf.placeholder(tf.int64, None)
resized_images = tf.image.resize_images(features, (227, 227))

# pass placeholder as first argument to `AlexNet`.
fc7 = AlexNet(resized_images, feature_extract=True)
# NOTE: `tf.stop_gradient` prevents the gradient from flowing backwards
# past this point, keeping the weights before and up to `fc7` frozen.
# This also makes training faster, less work to do!
fc7 = tf.stop_gradient(fc7)


# Add the final layer for traffic sign classification.
shape = (fc7.get_shape().as_list()[-1], nb_classes)
fc8W = tf.Variable(tf.truncated_normal(shape, stddev=1e-2))
fc8B = tf.Variable(tf.zeros(nb_classes))
logits = tf.nn.xw_plus_b(fc7, fc8W, fc8B)


# Define loss, training, accuracy operations.
# Distance vector between predictions and labels
cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,
                                                               labels=labels)
loss_op = tf.reduce_mean(cross_entropy)
opt = tf.train.AdamOptimizer()
train_op = opt.minimize(loss_op, var_list=[fc8W, fc8B])
init_op = tf.initialize_all_variables()

predictions = tf.arg_max(logits, 1)
accuracy_op = tf.reduce_mean(tf.cast(tf.equal(predictions, labels), tf.float32))


# Train and evaluate the feature extraction model.
def validate_model(X, y, sess):
    total_accuracy = 0
    total_loss = 0
    for start in range(0, X.shape[0], batch_size):
        end = start + batch_size
        X_batch = X[start:end]
        y_batch = y[start:end]

        loss, acc = sess.run([loss_op, accuracy_op], 
                             feed_dict={features: X_batch, labels: y_batch})

        total_accuracy += acc * X_batch.shape[0]
        total_loss += loss * X_batch.shape[0]
    return total_loss / X.shape[0], total_accuracy / X.shape[0]

with tf.Session() as sess:
    sess.run(init_op)

    for i in range(epochs):
        X_train, y_train = shuffle(X_train, y_train)
        t0 = time.time()
        for start in range(0, X_train.shape[0], batch_size):
            end = start + batch_size
            sess.run(train_op, feed_dict={features: X_train[start:end], 
                                          labels: y_train[start:end]})

        val_loss, val_acc = validate_model(X_val, y_val, sess)
        print("Epoch", i + 1)
        print("Time: %.3f seconds" % (time.time() - t0))
        print("Validation Loss =", val_loss)
        print("Validation Accuracy =", val_acc)
        print("")
