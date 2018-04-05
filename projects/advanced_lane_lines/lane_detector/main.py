import cv2
import matplotlib.image as mpimg
from moviepy.editor import VideoFileClip

import os
import sys

from .utils import CameraCalibration, LaneDetector, PerspectiveTransformation
from .utils import fs_utils

DATA_ZIP = 'https://s3-us-west-1.amazonaws.com/carnd-data/advanced_lane_lines_data.zip'
BASE_DOWNLOAD_DIR = './data/'
TEST_DATA_ZIP = './data/advanced_lane_lines_data.zip'

# Location to find camera parameters
CAMERA_OUTPUT_PARAMS_FILE = os.path.join(BASE_DOWNLOAD_DIR, 'camera_cal/output_params.json')

# Chess board settings
CHESS_BOARD_COLS = 9
CHESS_BOARD_ROWS = 6
CHESS_BOARD_IMAGES_PATH = os.path.join(BASE_DOWNLOAD_DIR, 'camera_cal/')

# TEST IMAGES PATH
TEST_IMAGES_DIR = os.path.join(BASE_DOWNLOAD_DIR, 'test_images/')
OUTPUT_IMAGES_DIR = os.path.join(BASE_DOWNLOAD_DIR, 'output_images/')

# VIDEO CLIP PATH
PROJECT_VIDEO = os.path.join(BASE_DOWNLOAD_DIR, 'project_video.mp4')
PROJECT_VIDEO_OUTPUT_DIR = os.path.join(BASE_DOWNLOAD_DIR, 'output_videos/')


def init_data():
    ''' 
    Initializes data required for the project.
    1. Downloads test data. 2. Runs camera calibration & Returns calibration object
    '''

    try:
        if not os.path.exists(TEST_DATA_ZIP):
            fs_utils.download_file(DATA_ZIP, BASE_DOWNLOAD_DIR)
            fs_utils.extract_zip(TEST_DATA_ZIP, BASE_DOWNLOAD_DIR)
    except Exception:
        # Clean up partial files
        if os.path.exists(TEST_DATA_ZIP):
            os.remove(TEST_DATA_ZIP)
        raise

    if not os.path.exists(OUTPUT_IMAGES_DIR):
        os.makedirs(OUTPUT_IMAGES_DIR)

    if not os.path.exists(PROJECT_VIDEO_OUTPUT_DIR):
        os.makedirs(PROJECT_VIDEO_OUTPUT_DIR)

    cc = None
    try:
        cc = CameraCalibration(params_load_path=CAMERA_OUTPUT_PARAMS_FILE)
    except ValueError:
        cc = CameraCalibration(CHESS_BOARD_COLS, CHESS_BOARD_ROWS, CHESS_BOARD_IMAGES_PATH)
    return cc


def process_pictures():

    cc = init_data()    
    mtx, dist = cc.get_camera_params()
    pp = PerspectiveTransformation()
    detector = LaneDetector(cc, pp, process_stream=False)

    test_images = [os.path.join(TEST_IMAGES_DIR, i) for i in os.listdir(TEST_IMAGES_DIR)]
    frames = [mpimg.imread(i) for i in test_images]
    for i, image in enumerate(frames):
        final_image = detector.process_frame(image)
        image_name = os.path.join(OUTPUT_IMAGES_DIR, os.path.basename(test_images[i]))
        final_image = cv2.cvtColor(final_image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(image_name, final_image)


def process_video():

    cc = init_data()
    mtx, dist = cc.get_camera_params()
    pp = PerspectiveTransformation()
    detector = LaneDetector(cc, pp, process_stream=True)

    output_video_path = os.path.join(PROJECT_VIDEO_OUTPUT_DIR,
                                     os.path.basename(PROJECT_VIDEO))
    clip = VideoFileClip(PROJECT_VIDEO)
    output_clip = clip.fl_image(lambda x: detector.process_frame(x))
    output_clip.write_videofile(output_video_path)


def main(argv):
    if len(argv) < 2:
        sys.exit('Usage python main.py pictures | video')

    if argv[1] == 'pictures':
        process_pictures()
    if argv[1] == 'video':
        process_video()

if __name__ == '__main__':
    main(sys.argv)
