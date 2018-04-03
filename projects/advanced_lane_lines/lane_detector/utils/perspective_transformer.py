import cv2

import numpy as np


class PerspectiveTransform(object):
    def __init__(self, source_points=None, destination_points=None, image=None):
        '''
        source_points & destination_points(2D Float32 List) list of 2D points for which
        the homography matrix shoulbe calculated
        TODO : Points are hardcoded right now. This can be improved by taking in an image
        and modifying the _calculate_tranformation() method by using HoughLines /Vanishing points
        as a starting point
        '''

        self.src_pts = source_points
        self.dst_pts = destination_points
        self.M = None
        self.M_inv = None

        offset_pix = 100  # Padding around the lane lines for the warped image

        if not self.src_pts:
            # TODO : Points found manually using images in test_images/straight_line1.jpg
            # This can be improved by building the set of points by calculating hough
            # lines or other techniques.

            y_bottom = 700
            y_horizon = 470
            n_cols = 1280  # hardcoded image size for Udacity Project
            self.src_pts = np.array([[offset_pix, y_bottom],  # Bottom Left
                                     [n_cols - offset_pix, y_bottom],  # Bottom Right
                                     [n_cols / 2 + offset_pix, y_horizon],  # Top right
                                     [n_cols / 2 - offset_pix, y_horizon]  # Top Left
                                     ], dtype=np.float32)

        if not self.dst_pts:
            dst_n_rows, dst_n_cols = 450, 800
            self.dst_pts = np.array([[offset_pix, dst_n_rows],  # Bottom Left
                                     [dst_n_cols - offset_pix, dst_n_rows],  # Bottom Right
                                     [dst_n_cols - offset_pix, 0],  # Top Right
                                     [offset_pix, 0],  # Top Left
                                     ], dtype=np.float32)

        self.__calculate_transformation()

    def __calculate_transformation(self):
        self.M = cv2.getPerspectiveTransform(self.src_pts, self.dst_pts)
        self.M_inv = cv2.getPerspectiveTransform(self.dst_pts, self.src_pts)

    def warp_image(self, image, warped_image_shape=None):
        if not warped_image_shape:
            warped_image_shape = (image.shape[1], image.shape[0])
        return cv2.warpPerspective(image, self.M, warped_image_shape, flags=cv2.INTER_LINEAR)

    def unwarp_image(self, warped_image, unwarped_image_shape=None):
        if not unwarped_image_shape:
            unwarped_image_shape = (warped_image.shape[1], warped_image.shape[0])
        return cv2.warpPerspective(warped_image, self.M_inv, unwarped_image_shape,
                                   flags=cv2.INTER_LINEAR)

    def get_destination_image_size(self):
        # TODO Clean this up, if using this class out side of the project
        return (450, 800)
