import json
import logging
import os
import pickle
import sys

import cv2

from moviepy.editor import VideoFileClip
from vehicle_detector.descriptors import HOG
from vehicle_detector.object_detection import ObjectDetector
from vehicle_detector.utils import fs_utils


logging.basicConfig(level='INFO')
logger = logging.getLogger('VehicleDetector')

DATA_ZIP = 'https://s3-us-west-1.amazonaws.com/carnd-data/advanced_lane_lines_data.zip'
TEST_DATA_ZIP = './data/advanced_lane_lines_data.zip'

BASE_DATA_DIR = './data/'
MODEL_PARAMS = './model/model_params.json'


# TEST IMAGES PATH
TEST_IMAGES_DIR = os.path.join(BASE_DATA_DIR, 'test_images/')
OUTPUT_IMAGES_DIR = os.path.join(BASE_DATA_DIR, 'output_images/')

# VIDEO CLIP PATH
PROJECT_VIDEO = os.path.join(BASE_DATA_DIR, 'project_video.mp4')
PROJECT_VIDEO_OUTPUT_DIR = os.path.join(BASE_DATA_DIR, 'output_videos/')


def init_detector(is_stream=False):

    try:
        if not os.path.exists(TEST_DATA_ZIP):
            fs_utils.download_file(DATA_ZIP, BASE_DATA_DIR)
            fs_utils.extract_zip(TEST_DATA_ZIP, BASE_DATA_DIR)
    except Exception:
        # Clean up partial files
        if os.path.exists(TEST_DATA_ZIP):
            os.remove(TEST_DATA_ZIP)
        raise

    if not os.path.exists(OUTPUT_IMAGES_DIR):
        os.makedirs(OUTPUT_IMAGES_DIR)

    if not os.path.exists(PROJECT_VIDEO_OUTPUT_DIR):
        os.makedirs(PROJECT_VIDEO_OUTPUT_DIR)

    # Load the picked model params
    with open(MODEL_PARAMS, 'r') as fp:
        params = json.load(fp)

    # Load the pickled Feature Scaler
    with open(params['scaler_file_path'], 'rb') as fp:
        scaler = pickle.loads(fp.read())

    # Load the pickled SVM Model
    with open(params['file_path'], 'rb') as fp:
        clf = pickle.loads(fp.read())

    hog_descriptor = HOG(params['orientations'], params['pixels_per_cell'],
                         params['cells_per_block'])
    detector = ObjectDetector(clf, scaler, hog_descriptor, params['color_hist'],
                              params['hog_color_space'], ystart=350, ystop=650,
                              is_stream=is_stream)
    return detector


def process_pictures():

    detector = init_detector()

    if not os.path.exists(OUTPUT_IMAGES_DIR):
        os.makedirs(OUTPUT_IMAGES_DIR)

    if not os.path.exists(TEST_IMAGES_DIR):
        os.makedirs(TEST_IMAGES_DIR)

    test_images = [os.path.join(TEST_IMAGES_DIR, i) for i in os.listdir(TEST_IMAGES_DIR)]
    frames = [cv2.imread(i) for i in test_images]
    for i, image in enumerate(frames):
        final_image = detector.process_frame(image)
        image_name = os.path.join(OUTPUT_IMAGES_DIR, os.path.basename(test_images[i]))
        # final_image = cv2.cvtColor(final_image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(image_name, final_image)


def process_video(video=None, tstart=None, tend=None):
    detector = init_detector(is_stream=True)

    if not os.path.exists(PROJECT_VIDEO_OUTPUT_DIR):
        os.makedirs(PROJECT_VIDEO_OUTPUT_DIR)
    if not video:
        video = PROJECT_VIDEO

    output_video_path = os.path.join(PROJECT_VIDEO_OUTPUT_DIR,
                                     os.path.basename(video))
    clip = VideoFileClip(video, audio=False)
    if tstart and tend:
        clip = clip.subclip(tstart, tend)

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
