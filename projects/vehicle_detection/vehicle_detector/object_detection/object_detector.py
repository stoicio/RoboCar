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

    def detect(self, image, pyramid_scale=1.5, min_prob=0.7):
        # initialize the list of bounding boxes and associated probabilities
        boxes = []
        probs = []

        if not self.ystart: 
            self.ystart = 0

        if not self.ystop:
            self.ystop = image.shape[0]

        img_tosearch = image[self.ystart:self.ystop, :, :]

        # loop over the image pyramid
        for layer in image_utils.pyramid(img_tosearch, scale=pyramid_scale,
                                         min_size=self.window_dim):
            # determine the current scale of the pyramid
            scale = img_tosearch.shape[0] / float(layer.shape[0])
            # loop over the sliding windows for the current pyramid layer
            for (x, y, window) in image_utils.sliding_window(layer, self.window_dim):
                # grab the dimensions of the window
                (winH, winW) = window.shape[:2]

                # ensure the window dimensions match the supplied sliding window dimensions
                if winH == self.window_dim[1] and winW == self.window_dim[0]:
                    # extract HOG features from the current window and classifiy whether or
                    # not this window contains an object we are interested in
                    features = self.extract_features(window).reshape(1, -1)
                    scaled_features = self.scaler.transform(features)
                    predict = self.model.predict(scaled_features) #_proba(scaled_features)[0][1]
                    
                    # check to see if the classifier has found an object with sufficient
                    # probability
                    if predict == 1:
                        # compute the (x, y)-coordinates of the bounding box using the current
                        # scale of the image pyramid
                        (startx, starty) = (int(scale * x), int(scale * y))
                        end_x = int(startx + (scale * winW))
                        end_y = int(starty + (scale * winH))

                        # update the list of bounding boxes and probabilities
                        boxes.append((startx, starty + self.ystart, end_x, end_y + self.ystart))
                        probs.append(predict)

        # return a tuple of the bounding boxes and probabilities
        return (boxes, probs)

    def extract_features(self, img):
        '''
        Given an image and options, extract the features and return a feature vector
        Args:
        img (np.array)  : Single or multichannel image
        use_color_hist (bool) : Set to True, to extract color histogram of each channel
        hog_cspace (str): String describing the color space to use to extract HOG features
        Returns:
        1D Feature vector describing the image
        '''
        win_size = self.window_dim[0]
        if img.shape[0] != win_size or img.shape[1] != win_size:
            img = image_utils.resize(img, win_size, win_size)

        color_features = []

        if self.extract_color_hist and len(img.shape) > 2:
            color_features = color_hist(img, nbins=32)

        if self.hog_cspace != 'BGR':
            img = image_utils.convert_color(img, self.hog_cspace)

        hog_features = self.hog.get_features(img, feature_vec=True)

        if len(img.shape) > 2:  # If multichannel image, concatenate all channel's features
            hog_features = np.concatenate((hog_features[0], hog_features[1], hog_features[2]))

        return np.concatenate((hog_features, color_features))
