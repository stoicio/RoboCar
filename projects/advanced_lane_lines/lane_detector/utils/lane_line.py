import logging

import numpy as np

logger = logging.getLogger('LaneLine')

Y_MTRS_PER_PIXEL = 30 / 720  # meters per pixel in y direction
X_MTRS_PER_PIXEL = 3.7 / 700  # meters per pixel in x direction


class LaneLine(object):
    def __init__(self, img_rows, img_cols, process_stream=False):

        self.n_rows = img_rows
        self.n_cols = img_cols
        self.image_h_center = self.n_cols // 2

        # Generate y coordinates that will be later used to fit a polynomial
        self.y_coordinates = np.linspace(0, self.n_rows - 1, self.n_rows)

        self.curr_fit_coef_left = None
        self.prev_fit_coef_left = []

        self.curr_fit_coef_right = None
        self.prev_fit_coef_right = []

        self.curr_curvature = None
        self.prev_curvature = []

        self.curr_lane_offset = None

        self.weights = np.float32([0.1, 0.2, 0.4, 0.6, 0.8]).reshape(5, 1)
        self.moving_average_window_size = 5

        self.is_process_stream = process_stream

    def __get_lane_pixels(self, image, margin=40):
        ''' Extract left and right lane pixels
        image - 8 bit single binary image, with lane pixels identified with value of 255
        margin - number of pixels around the base of the lane line to search for
                margin = 75 will look for at 150 pixels around the lane base
        '''
        # Count white pixels in each col of the bottom half of the image
        histogram = np.sum(image[self.n_rows // 2:, :], axis=0)

        left_base = np.argmax(histogram[:self.image_h_center])
        right_base = np.argmax(histogram[self.image_h_center:]) + self.image_h_center

        # Extract columns surronding the left and the right of the lane base
        left_min = max(left_base - margin, 0)
        left_max = min(left_base + margin, self.image_h_center)

        right_min = max(right_base - margin, self.image_h_center)
        right_max = min(right_base + margin, image.shape[1])

        left_lane_roi = image[:, left_min:left_max]
        right_lane_roi = image[:, right_min:right_max]

        left_indices_y, left_indices_x = np.where(left_lane_roi == 255)
        left_indices_x += left_min  # Add offset back

        right_indices_y, right_indices_x = np.where(right_lane_roi == 255)
        right_indices_x += right_min  # Add offset back

        return ((left_indices_y, left_indices_x), (right_indices_y, right_indices_x))

    def __calculate_radii_of_curvature(self, left_lane_pixels, right_lane_pixels):
        '''
        Returns the last calculated radius of curvature in metres

        if processing a stream, returns the SOME_AVERAGE of previously
        measured values and current value
        '''
        # Convert from pixel to world coordinates
        left_world_coord = (left_lane_pixels[0] * Y_MTRS_PER_PIXEL,
                            left_lane_pixels[1] * X_MTRS_PER_PIXEL)

        right_world_coord = (right_lane_pixels[0] * Y_MTRS_PER_PIXEL,
                             right_lane_pixels[1] * X_MTRS_PER_PIXEL)

        y_evaluate = np.max(left_lane_pixels[0])  # Choose the bottom most part of the lane

        # Fit new polynomials to x,y in world space
        left_fit = self.__get_best_fit_coefs(left_world_coord)
        right_fit = self.__get_best_fit_coefs(right_world_coord)

        # Calculate radii of curvature
        left_curvature = ((1 + (2 * left_fit[0] * y_evaluate * Y_MTRS_PER_PIXEL +
                                left_fit[1])**2) ** 1.5) / np.absolute(2 * left_fit[0])

        right_curvature = ((1 + (2 * right_fit[0] * y_evaluate * Y_MTRS_PER_PIXEL +
                                 right_fit[1])**2) ** 1.5) / np.absolute(2 * right_fit[0])
        curr_curvature = np.mean([left_curvature, right_curvature])

        return round(curr_curvature, 2)

    def __get_best_fit_coefs(self, pixel_indices, degree_of_polynomial=2):
        y, x = pixel_indices
        return np.polyfit(y, x, deg=degree_of_polynomial)

    def __get_best_fit_line(self, poly_coefs, degree=2):
        best_fit_x = (poly_coefs[0] * self.y_coordinates ** 2 +
                      poly_coefs[1] * self.y_coordinates + poly_coefs[2])
        return best_fit_x

    def __calculate_lane_offset(self, bottom_left_pixel, bottom_right_pixel):
        car_h_center = (bottom_left_pixel[0] + bottom_right_pixel[0]) / 2
        offset = (self.image_h_center - car_h_center) * X_MTRS_PER_PIXEL
        return round(offset, 2)

    def extract_lane_lines(self, binary_image):
        '''
        Given a binary image, which has the lane pixels in white color
        extracts the left & right lane pixels and fit a polynomial curve
        to it
        binary_image (np.array) single channel 8 bit unsigned array, where all extracted
                                lane pixels have value 255
        '''
        # Extract indices of left and right lane pixels
        left_lane_pixels, right_lane_pixels = self.__get_lane_pixels(binary_image)

        calculated_left_fit = self.__get_best_fit_coefs(left_lane_pixels)
        calculated_right_fit = self.__get_best_fit_coefs(right_lane_pixels)

        calculated_curvature = self.__calculate_radii_of_curvature(left_lane_pixels,
                                                                   right_lane_pixels)

        if not self.is_process_stream:
            self.curr_fit_coef_left = calculated_left_fit
            self.curr_fit_coef_right = calculated_right_fit
            self.curr_curvature = calculated_curvature
        elif self.is_process_stream and (len(self.prev_fit_coef_left) <
                                         self.moving_average_window_size):
            # Have not got enough frame to do moving average yet
            self.curr_fit_coef_left = calculated_left_fit
            self.curr_fit_coef_right = calculated_right_fit
            self.curr_curvature = calculated_curvature
            self.prev_fit_coef_left.append(calculated_left_fit.tolist())
            self.prev_fit_coef_right.append(calculated_right_fit.tolist())
            self.prev_curvature.append(calculated_curvature)
        else:  # Have enough data to do smoothing
            # Remove oldest values
            self.prev_curvature = self.prev_curvature[1:]
            self.prev_fit_coef_left = self.prev_fit_coef_left[1:]
            self.prev_fit_coef_right = self.prev_fit_coef_right[1:]
            # Add new values
            self.prev_curvature.append(calculated_curvature)
            self.prev_fit_coef_left.append(calculated_left_fit.tolist())
            self.prev_fit_coef_right.append(calculated_right_fit.tolist())
            # Calculate weighted average

            self.curr_fit_coef_left = np.sum(np.float32(
                                             self.prev_fit_coef_left) * self.weights, axis=0
                                             ) / np.sum(self.weights)
            self.curr_fit_coef_right = np.sum(np.float32(
                                              self.prev_fit_coef_right) * self.weights, axis=0
                                              ) / np.sum(self.weights)
            self.curr_curvature = np.sum(np.float32(
                                         self.prev_curvature) * self.weights.squeeze()
                                         ) / np.sum(self.weights)

        best_left_line_x = self.__get_best_fit_line(self.curr_fit_coef_left)
        best_right_line_x = self.__get_best_fit_line(self.curr_fit_coef_right)

        left_line_to_draw = np.int32(list(zip(best_left_line_x, self.y_coordinates)))
        right_line_to_draw = np.int32(list(zip(best_right_line_x, self.y_coordinates)))

        self.curr_lane_offset = self.__calculate_lane_offset(left_line_to_draw[-1],
                                                             right_line_to_draw[-1])
        return {
            'left_lane': left_line_to_draw,
            'right_lane': right_line_to_draw,
            'curvature': self.curr_curvature,
            'offset': self.curr_lane_offset
        }
