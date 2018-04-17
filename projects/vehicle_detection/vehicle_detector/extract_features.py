import logging

import cv2
import numpy as np
from tqdm import tqdm

from .descriptors import HOG, color_hist
from .utils import dataset, image_utils

logger = logging.getLogger(__name__)


def extract_features(img, use_color_hist, hog_descriptor, hog_cspace):
    '''
    Given an image and options, extract the features and return a feature vector
    Args:
     img (np.array)  : Single or multichannel image
     use_color_hist (bool) : Set to True, to extract color histogram of each channel
     hog_cspace (str): String describing the color space to use to extract HOG features
    Returns:
     1D Feature vector describing the image
    '''
    color_features = []

    if hog_cspace != 'BGR':
        img = image_utils.convert_color(img, hog_cspace)

    if use_color_hist and len(img.shape) > 2:
        color_features = color_hist(img, nbins=32)

    hog_features = hog_descriptor.get_features(img, feature_vec=True)

    if len(img.shape) > 2:  # If multichannel image, concatenate all channel's features
        hog_features = np.concatenate((hog_features[0], hog_features[1], hog_features[2]))

    return np.concatenate((hog_features, color_features))


def get_feature_vectors(image_paths, use_color_hist, orients, n_pixels, n_cells, hog_cspace):
    '''
    Given a list of image paths, read each image and extract requested fetures
    Args:
    use_color_hist (bool) : Set to True, to extract color histogram of each channel
    orients (int) : Number of Orientation bins to use for each cell (HOG Parameter)
    n_pixels (tuple) : Number of pixels to use in each cell (HOG Parameter)
    n_cells (tuple): Number of cells to normalize over (HOG Parameter)
    hog_cspace (str): String describing the color space to use to extract HOG features
    Returns:
    List of feature vectors describing all the images
    '''

    features = []

    # Create HOG Descriptor with given parameters
    hog_descriptor = HOG(orients, n_pixels, n_cells)

    for img_path in tqdm(image_paths, desc='Extracting features', unit='images', leave=False):

        img = cv2.imread(img_path)

        if img.shape[0] != 64 or img.shape[1] != 64:
            img = cv2.resize(img, (64, 64))

        features.append(extract_features(img, use_color_hist, hog_descriptor, hog_cspace))
    return features


def extract_dataset_features(positive_image_paths, negative_image_paths, params):
    '''
    Extracts the features from a given list of positive and negative image samples
    with the given params and returns a tuple of features and labels
    Args:
    *_image_paths (list) - list of paths
    params(dict) - describes the features to extract. Keys as follows
    params['color_hist] (bool) : Set to True, to extract color histogram of each channel
    params['orientations'] (int) : Number of Orientation bins to use for each cell (HOG Parameter)
    params['pixels_per_cell'] (tuple) : Number of pixels to use in each cell (HOG Parameter)
    params['cells_per_block'] (tuple): Number of cells to normalize over (HOG Parameter)
    params['hog_color_space'] (str): Describes the color space to use to extract HOG features from
    '''
    car_features = get_feature_vectors(positive_image_paths, params['color_hist'],
                                       params['orientations'], params['pixels_per_cell'],
                                       params['cells_per_block'], params['hog_color_space'])
    non_car_features = get_feature_vectors(negative_image_paths, params['color_hist'],
                                           params['orientations'], params['pixels_per_cell'],
                                           params['cells_per_block'], params['hog_color_space'])

    n_car_samples = len(positive_image_paths)
    n_non_car_samples = len(negative_image_paths)

    features = np.vstack((car_features, non_car_features)).astype(np.float64)
    labels = np.hstack((np.ones(n_car_samples), np.zeros(n_non_car_samples)))

    return (features, labels)


def extract_save_features(positive_image_paths, negative_image_paths, params,
                          dataset_path, dataset_name):
    '''
    Extracts the features from a given list of positive and negative image samples
    with the given params, and save them to a HDF5 dataset with a specific name
    Args:
    *_image_paths (list) - list of paths

    params(dict) - describes the features to extract. Keys as follows
    params['color_hist] (bool) : Set to True, to extract color histogram of each channel
    params['orientations'] (int) : Number of Orientation bins to use for each cell (HOG Parameter)
    params['pixels_per_cell'] (tuple) : Number of pixels to use in each cell (HOG Parameter)
    params['cells_per_block'] (tuple): Number of cells to normalize over (HOG Parameter)
    params['hog_color_space'] (str): String describing color space to use to extract HOG features

    dataset_path (str) : Path to create a new HDF5 dataset
    dataset_name (str) : Name of the dataset (since one HDF5 can contain many datasets)
    '''
    features, labels = extract_dataset_features(positive_image_paths, negative_image_paths, params)
    dataset.write_dataset(features, labels, dataset_path, dataset_name)
    logger.info('Successfully wrote dataset to file at %s with name %s', dataset_path,
                dataset_name)
    return (dataset_path, dataset_name)
