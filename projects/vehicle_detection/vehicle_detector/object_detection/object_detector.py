import numpy as np
from vehicle_detector.descriptors import color_hist
from vehicle_detector.utils import image_utils


class ObjectDetector:
    def __init__(self, model, feature_scaler, hog_descriptor, color_hist,
                 hog_cspace='BGR', ystart=None, ystop=None, window_dim=(64, 64)):
        # Initialize with classifier and options for feature extraction
        self.model = model  # Classifier
        self.scaler = feature_scaler  # Sklearn standard scaler
        self.hog = hog_descriptor  # Hog descriptor with colors loaded
        self.extract_color_hist = color_hist  # Boolean, extract colot histogram if true
        self.hog_cspace = hog_cspace  # Color space to use for HOG features
        # ystart & ystop defines the portion of the image used for scanning for cars
        self.ystart = ystart
        self.ystop = ystop
        self.window_dim = window_dim

        # Store hog_descriptor params
        self.orientations = self.hog.orient
        self.pixels_per_cell = self.hog.pix_per_cell[0]
        self.cells_per_block = self.hog.cells_per_block[0]

    def detect(self, image, scales=[1., 1.25, 1.5, 1.75, 2.0]):
        boxes = []
        probs = []

        for scale in scales:
            this_boxes, this_probs = self.find_cars(image, scale)
            boxes.extend(this_boxes)
            probs.extend(this_probs)
        return boxes, probs

    # Define a single function that can extract features using hog sub-sampling and make predictions
    def find_cars(self, image, scale=1.0, min_prob=0.8):
        # initialize the list of bounding boxes and associated probabilities
        boxes = []
        probs = []

        if not self.ystart:
            self.ystart = 0

        if not self.ystop:
            self.ystop = image.shape[0]

        img_tosearch = image[self.ystart:self.ystop, :, :]

        ctrans_tosearch = img_tosearch
        if self.hog_cspace != 'BGR':
            ctrans_tosearch = image_utils.convert_color(img_tosearch, conv=self.hog_cspace)

        if scale != 1:
            imshape = ctrans_tosearch.shape
            ctrans_tosearch = image_utils.resize(ctrans_tosearch, width=np.int(imshape[1] / scale),
                                                 height=np.int(imshape[0] / scale))

        # Define blocks and steps as above
        nxblocks = (ctrans_tosearch.shape[1] // self.pixels_per_cell) - self.cells_per_block + 1
        nyblocks = (ctrans_tosearch.shape[0] // self.pixels_per_cell) - self.cells_per_block + 1

        # 64 was the orginal sampling rate, with 8 cells and 8 pix per cell
        window = 64  # TODO Assert if window dims are divisible by pixels/cell and cells/block
        nblocks_per_window = (window // self.pixels_per_cell) - self.cells_per_block + 1
        cells_per_step = 2  # Instead of overlap, define how many cells to step
        nxsteps = (nxblocks - nblocks_per_window) // cells_per_step + 1
        nysteps = (nyblocks - nblocks_per_window) // cells_per_step + 1

        # Compute individual channel HOG features for the entire image
        (hog1, hog2, hog3) = self.hog.get_features(ctrans_tosearch, feature_vec=False)

        for xb in range(nxsteps):
            for yb in range(nysteps):
                ypos = yb * cells_per_step
                xpos = xb * cells_per_step
                # Extract HOG for this patch
                hog_feat1 = hog1[ypos:ypos + nblocks_per_window,
                                 xpos:xpos + nblocks_per_window].ravel()
                hog_feat2 = hog2[ypos:ypos + nblocks_per_window,
                                 xpos:xpos + nblocks_per_window].ravel()
                hog_feat3 = hog3[ypos:ypos + nblocks_per_window,
                                 xpos:xpos + nblocks_per_window].ravel()
                hog_features = np.concatenate((hog_feat1, hog_feat2, hog_feat3))

                xleft = xpos * self.pixels_per_cell
                ytop = ypos * self.pixels_per_cell

                # Extract the image patch
                image_patch = ctrans_tosearch[ytop:ytop + window, xleft:xleft + window]
                subimg = image_utils.resize(image_patch, width=64, height=64)

                # Get color features
                color_features = []
                if self.extract_color_hist:
                    color_features = color_hist(subimg)

                features = np.concatenate((hog_features, color_features)).reshape(1, -1)

                # Scale features and make a prediction
                test_features = self.scaler.transform(features).reshape(1, -1)
                # Get probability for  1st label, Car in this case
                probability = self.model.predict_proba(test_features)[0][1] 

                if probability >= min_prob:
                    xbox_left = np.int(xleft * scale)
                    ytop_draw = np.int(ytop * scale)
                    win_draw = np.int(window * scale)
                    boxes.append((xbox_left, ytop_draw + self.ystart,
                                  xbox_left + win_draw, ytop_draw + win_draw + self.ystart))
                    probs.append(probability)
        return boxes, probs
