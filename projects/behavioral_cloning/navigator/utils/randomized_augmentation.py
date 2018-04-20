import cv2
import numpy as np

from . import image_utils


def h_flip(image, steering_angle):
    '''Given an image, flip it around the vertical axis randomly'''
    if np.random.choice([0, 1]) == 1:
        # Flip along vertical axis
        image, steering_angle = cv2.flip(image, flipCode=1), -1 * steering_angle
    return image, steering_angle


def steering_correction(cam_position, curr_angle, recovery_distance=4.0, cam_offset=1.0):
    '''
    Given a steering angle and a target camera_position (left or right), correct the
    given steering angle to account for the camera offset.

    recovery_distance (float) : Distance within which the car must correct its position to
                               reach the center of the lane (in this case the original
                               steering angle)
    cam_offset (float): Distance between the center and one of the side cameras
    '''
    correction = cam_offset / recovery_distance
    sign = 0
    if cam_position == 'left':
        sign = 1
    if cam_position == 'right':
        sign = -1
    return curr_angle + (correction * sign)


def choose_cam_position(img_path, steering_angle, recovery_distance=4.0, cam_offset=1.0):

    '''
    Given a center camera image path and steering angle, randomly choose if we should
    use that image or the corresponding left or right image. Adjust steering angle based
    on the choosen camera position.

    recovery_distance (float) : Distance within which the car must correct its position to
                               reach the center of the lane (in this case the original
                               steering angle)
    cam_offset (float): Distance between the center and one of the side cameras
    '''
    position = np.random.choice(['center', 'left', 'right'])

    if position != 'center':
        img_path = img_path.replace('center', position)
        steering_angle = steering_correction(position, steering_angle, recovery_distance,
                                             cam_offset)
    return img_path, steering_angle


def remove_zero_angle_logs(img_paths, steering_angles, retention_rate=0.1):
    '''
    Given X & Y values for each row in the telemetry data, retain only 10%(or given rate)
    of the data with 0 steering angle and delete other entries'''
    zero_indices = np.where(steering_angles == 0.0)[0]
    # Choose indices to delete, set replace=False, which would otherwise include
    # repeated indices
    indices_to_remove = np.random.choice(zero_indices,
                                         size=np.int((1.0 - retention_rate) * len(zero_indices)),
                                         replace=False)
    img_paths = np.delete(img_paths, indices_to_remove, axis=0)
    steering_angles = np.delete(steering_angles, indices_to_remove, axis=0)
    return img_paths, steering_angles


def random_gamma_adjust(image, steering_angle, gamma_range=[0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75,
                                                            2.0]):
    '''
    Given an image array, adjust its brightness by choosing a value from a range'''
    gamma = np.random.choice(gamma_range)  # Choose a gamma value to adjust brightness by
    return image_utils.adjust_gamma(image, gamma), steering_angle


def read_and_augument_image(image_path, steering_angle, model_input_size=(160, 320)):
    # Choose which camera's image to use
    image_path, steering_angle = choose_cam_position(image_path, steering_angle)
    # Read in the image to memory
    image = cv2.imread(image_path)
    image, steering_angle = random_gamma_adjust(image, steering_angle)
    image, steering_angle = h_flip(image, steering_angle)
    return image, steering_angle
