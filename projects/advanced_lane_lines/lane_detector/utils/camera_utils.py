import json
import logging
import os

import cv2
import numpy as np
from tqdm import tqdm

logger = logging.getLogger('CameraUtils')


class CameraCalibration(object):

    @staticmethod
    def get_image_paths(chessboard_img_dir):
        allowed_extensions = ['.jpg', '.png', '.jpeg']
        full_image_paths = []
        if not os.path.exists(chessboard_img_dir) or os.path.isfile(chessboard_img_dir):
            raise ValueError("Chessboard images directory not found")
        files_in_dir = os.listdir(chessboard_img_dir)
                
        for _file in files_in_dir:
            if os.path.splitext(_file)[-1] in allowed_extensions:
                full_image_paths.append(os.path.join(chessboard_img_dir, _file))
            else:
                logger.info("Skipping {name} - Not an image file".format(name=_file))

        if not full_image_paths:
            raise RuntimeError("No chessboard images found")
        return full_image_paths

    
    def __init__(self, n_cols=None, n_rows=None, chessboard_img_dir=None,
                 params_load_path=None, store_output_images=False,):
        '''
        Args:
            n_cols (int) : Number of corners along horizontal axis
            n_rows (int) : Number of corners along vertical axis
            chessboard_img_dir (str) : directory where the chessboard images are stored
        '''
        if params_load_path:
            if os.path.exists(params_load_path):
                self.load_params_from_file(params_load_path)
                logger.info('Camera params loaded and ready to use')
            else:
                logger.error('Cannot load params from file. Please recalibrate')
                raise ValueError('Cannot load params from file. Please recalibrate')
            return
    
        
        if not all([n_cols, n_rows, chessboard_img_dir]):
            raise ValueError('Pass in chess board params and location to images')
        
        
        self.images_dir = chessboard_img_dir
        self.image_paths = self.get_image_paths(chessboard_img_dir)
        self.pattern_size = (n_cols, n_rows)

        self.mtx = None
        self.dist = None

        self.output_images_path = []
        self.failed_images = []
        self.__is_calibrated = False
        self.__store_output_images = store_output_images
        self.__calibrate_camera()

    def load_params_from_file(self, json_file_path):
        expected_keys = ['mtx', 'dist']
        with open(json_file_path, 'r') as fp:
            data = json.load(fp)
        if not all([k in data.keys() for k in expected_keys]):
            raise ValueError('Cannot load camera params. Use a different file or recalibrate')
        self.mtx = np.array(data['mtx'])
        self.dist = np.array(data['dist'])
        self.__is_calibrated = True
    
    def save_params_to_file(self, file_path):
        data = {
            'mtx': self.mtx.tolist(),
            'dist': self.dist.tolist()
        }
        with open(file_path, 'w') as fp:
            json.dump(data, fp)


    def __calibrate_camera(self):

        # Termination criteria to choose accurate corners. terminate sub-pixel detection 
        # after 30 iterations or if improvement is less than 0.001
        termination_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

        # Arrays to store collection of 3d and 2d chessboard corners
        chessboard_corners_3d = []
        image_points_2d = []

        corner_points_3d = np.zeros((self.pattern_size[0] * self.pattern_size[1], 3), np.float32)
        # Fill with 3D Coordinates representing the corners in chess board
        corner_points_3d[:, :2] = np.mgrid[0: self.pattern_size[0], 0:self.pattern_size[1]].T.reshape(-1, 2)  # flake8: noqa

        # if we have to store output images of detected chess boards, create a target folder
        output_imgs_dir = os.path.join(self.images_dir, 'output')

        if self.__store_output_images and not os.path.exists(output_imgs_dir):
            os.makedirs(output_imgs_dir)

        for image in tqdm(self.image_paths, desc='Finding chessboard corners'):
            img = cv2.imread(image)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Find corners - return early if no corners are detectable
            found_corners, corners = cv2.findChessboardCorners(gray, self.pattern_size,
                                                               None,
                                                               cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK)

            if found_corners:
                chessboard_corners_3d.append(corner_points_3d)
                accurate_corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1),
                                                    termination_criteria)

                image_points_2d.append(accurate_corners)

                if self.__store_output_images:
                    new_img_path = os.path.join(output_imgs_dir, os.path.basename(image))
                    cv2.drawChessboardCorners(img, self.pattern_size, accurate_corners,
                                              found_corners)
                    cv2.imwrite(new_img_path, img)
                    self.output_images_path.append(new_img_path)
            else:
                logger.debug("Failed to find chessboard in {name}".format(name=image))
                self.failed_images.append(image)

        (success, self.mtx, self.dist, _, _) = cv2.calibrateCamera(chessboard_corners_3d,
            image_points_2d, gray.shape[::-1], None, None)

        if not success:
            raise RuntimeError("Calibration failed ! Retry with better chessboard images")
        
        # Set Calibration Result to Trues
        logger.info(('Successfully calculated Camera Matrix.'
                     'Skipped processing {count} images').format(count=len(self.failed_images)))
        self.__is_calibrated = True

    def get_camera_params(self, redo_calibration=False):
        if not self.__is_calibrated or redo_calibration:
            self.__calibrate_camera()
        return (self.mtx, self.dist)

    def get_processed_images(self):
        '''Returns a list of chessboard images with corners drawn and a list of images
        in which corner detection failed
        Returns data (dict):
            data['output_images'] : list of paths with corners drawn
            data['failed_images'] : list of path in which corner detection failed
        '''
        if not self.__store_output_images:
            logger.warn(('Output images are not stored. To write output images,'
                        'set "store_ store_output_images=True" during init'))
        return {
            'output_images': self.output_images_path,
            'failed_images': self.failed_images
        }

    def undistort_image(self, image):
        '''Takes an numpy array representing an image or a string pointing to a image path
        and undistorts with the calibrated camera matrix and distortion coffiecients'''
        
        if not self.__is_calibrated:
            self.__calibrate_camera()
        
        img_data = cv2.imread(image) if isinstance(image, str) else image
        return cv2.undistort(img_data, self.mtx, self.dist, None, self.mtx)
