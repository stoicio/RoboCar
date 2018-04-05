import cv2

from . import lane_line

import numpy as np


class LaneDetector(object):
    def __init__(self, camera_calibration, perspective_transform, process_stream=False):
        '''Build a lane detector object that can process a single frame
        or a stream of camera frames
        camera_calibration (object) - Object which has camera params loaded
                                    & helper functions to undistort images
        perpective_transfrom (object) - Object which has a homography matrix to
                                    transform a forward looking frame to a overhead view
        '''
        self.cam_util = camera_calibration
        self.transform_util = perspective_transform
        # self.is_process_stream = process_stream

        n_rows, n_cols = self.transform_util.get_destination_image_size()
        self.warped_img_shape = (n_cols, n_rows)

        self.small_kernel = np.ones((7, 7), np.uint8)
        self.lane_helper = lane_line.LaneLine(n_rows, n_cols, process_stream)

        self.count = 0

    def combine_s_v(self, image, min_val=155, max_val=255):
        '''
        Given a color image, converts it to HSV color scale,
        discards H channel and replaces with V channel and
        takes a mean of all three channels then applies a
        threshold to create a masked binary image
        '''
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

        s_channel = hsv[:, :, 1]
        v_channel = hsv[:, :, 2]

        v_channel = cv2.dilate(v_channel, self.small_kernel, iterations=2)

        combined = np.mean(np.stack((s_channel, v_channel, v_channel), axis=2), 2)
        combined = cv2.inRange(combined, min_val, max_val)

        return combined

    def draw_lane_lines(self, lane_data, original_image, warped_image):

        out_img = np.zeros_like(warped_image)

        left_line_to_draw = lane_data['left_lane']
        right_line_to_draw = lane_data['right_lane']

        center_line = np.int32((left_line_to_draw + right_line_to_draw) / 2)

        lane_mask = np.int32([left_line_to_draw[0],
                              right_line_to_draw[0],
                              right_line_to_draw[-1],
                              left_line_to_draw[-1]])

        cv2.fillPoly(out_img, [lane_mask], color=(0, 180, 0))
        cv2.polylines(out_img, [left_line_to_draw], False, color=(255, 0, 0), thickness=25)
        cv2.polylines(out_img, [right_line_to_draw], False, color=(0, 0, 255), thickness=25)
        cv2.polylines(out_img, [center_line], False, color=(255, 0, 255), thickness=5)

        unwarped_lane_mask = self.transform_util.unwarp_image(out_img, (original_image.shape[1],
                                                                        original_image.shape[0]))

        final_image = cv2.addWeighted(original_image, 1, unwarped_lane_mask, 0.8, 0)

        # Annotate lane curvature values and vehicle offset from center
        radius = 'Radius of curvature: {radius} m'.format(radius=lane_data['curvature'])
        cv2.putText(final_image, radius, (30, 40), 0, 1, (255, 255, 255), 4, cv2.LINE_AA)
        cv2.putText(final_image, radius, (30, 40), 0, 1, (0, 0, 0), 2, cv2.LINE_AA)

        offset = 'Off from lane center: {offset} m'.format(offset=lane_data['offset'])
        cv2.putText(final_image, offset, (30, 70), 0, 1, (255, 255, 255), 4, cv2.LINE_AA)
        cv2.putText(final_image, offset, (30, 70), 0, 1, (0, 0, 0), 2, cv2.LINE_AA)
        return final_image

    def process_frame(self, color_image_rgb):
        undistorted = self.cam_util.undistort_image(color_image_rgb)
        warped_image = self.transform_util.warp_image(undistorted, self.warped_img_shape)
        masked_s_v = self.combine_s_v(warped_image)
        try:
            lane_data = self.lane_helper.extract_lane_lines(masked_s_v)
            output_img = self.draw_lane_lines(lane_data, undistorted, warped_image)
        except Exception:
            self.count = self.count + 1
            failed_image = cv2.cvtColor(undistorted, cv2.COLOR_RGB2BGR)
            cv2.imwrite('./failed_image_{count}.jpg'.format(count=self.count), failed_image)
            return undistorted
        return output_img
