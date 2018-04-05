import cv2
import matplotlib.image as mpimg
from moviepy.editor import VideoFileClip

import os
import sys

from .utils import CameraCalibration, LaneDetector, PerspectiveTransformation

# Location to find camera parameters
CAMERA_OUTPUT_PARAMS_FILE = './camera_cal/output_params.json'

# Chess board settings
CHESS_BOARD_COLS = 9
CHESS_BOARD_ROWS = 6
CHESS_BOARD_IMAGES_PATH = './camera_cal'

# TEST IMAGES PATH
TEST_IMAGES_DIR = './test_images'
OUTPUT_IMAGES_DIR = './output_images'

# VIDEO CLIP PATH
PROJECT_VIDEO = './project_video.mp4'
PROJECT_VIDEO_OUTPUT_DIR = './output_videos'

if not os.path.exists(OUTPUT_IMAGES_DIR):
    os.makedirs(OUTPUT_IMAGES_DIR)

if not os.path.exists(PROJECT_VIDEO_OUTPUT_DIR):
    os.makedirs(PROJECT_VIDEO_OUTPUT_DIR)


def process_pictures():
    cc = CameraCalibration(params_load_path=CAMERA_OUTPUT_PARAMS_FILE)
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
    cc = CameraCalibration(params_load_path=CAMERA_OUTPUT_PARAMS_FILE)
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
