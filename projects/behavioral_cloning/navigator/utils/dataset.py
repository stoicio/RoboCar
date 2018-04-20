import cv2

import numpy as np
from sklearn.utils import shuffle

from . import image_utils, read_and_augument_image, remove_zero_angle_logs


def prep_image_for_model(image, model_image_size, y_start=80, y_end=135, colorspace='BGR2YUV'):
    # Crop image to ROI
    image = image[y_start:y_end, :, :]

    # Convert image to required colorspace
    image = image_utils.convert_color(image, colorspace)
    # Resize image to size the model expects
    if image.shape[0] != model_image_size[0] or image.shape[1] != model_image_size[1]:
        image = image_utils.resize(image, width=model_image_size[1], height=model_image_size[0])

    return image


def batch_generator(image_paths, steering_angles, model_image_size, batch_size=128,
                    training=True):

    def init_dataset():
        if training:
            X_pre, y_pre = shuffle(image_paths, steering_angles)
            X_pre, y_pre = remove_zero_angle_logs(X_pre, y_pre)
        else:
            X_pre, y_pre = image_paths, steering_angles
        return X_pre, y_pre

    # Have an offset to start over from intial data set if we run out of image during
    # batch generation
    batch_num = 0
    y_bag = []
    X_pre, y_pre = init_dataset()

    while True:
        X = np.empty((batch_size, model_image_size[0], model_image_size[1], model_image_size[2]))
        y = np.empty((batch_size, 1))

        for idx in range(batch_size):

            img_path, angle = X_pre[batch_num + idx], y_pre[batch_num + idx]

            if training:
                image, angle = read_and_augument_image(img_path, angle)
            else:  # No preprocessing for validation_data
                image = cv2.imread(img_path)

            X[idx, :, :, :] = prep_image_for_model(image, model_image_size)
            y[idx] = angle
            y_bag.append(y[idx])
            if (batch_num + idx) >= (y_pre.shape[0] - 1):
                X_pre, y_pre = init_dataset()
                batch_num = 0

        batch_num += batch_size
        yield X, y

        if training:
            np.save('y_bag.npy', np.array(y_bag))
            np.save('X_batch.npy', X)
