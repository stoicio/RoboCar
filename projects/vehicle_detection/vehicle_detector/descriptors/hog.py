from skimage.feature import hog


class HOG:
    def __init__(self, orientations=9, pixels_per_cell=(8, 8),
                 cells_per_block=(2, 2), normalize=True):
        self.orient = orientations
        self.pix_per_cell = pixels_per_cell
        self.cells_per_block = cells_per_block
        self.normalize = normalize

    def get_features_single_channel(self, img, feature_vec=True, vis=False,
                                    block_norm_type='L2-Hys'):
        '''
        Given a single channel image, extract HOG features from that image.
        '''
        output = hog(img, orientations=self.orient,
                     pixels_per_cell=self.pix_per_cell,
                     cells_per_block=self.cells_per_block,
                     block_norm=block_norm_type, transform_sqrt=self.normalize,
                     visualise=vis, feature_vector=feature_vec)

        if vis:
            features, hog_image = output
            return features, hog_image
        else:
            return output

    def get_features(self, img, feature_vec=True, vis=False, channel='ALL',
                     block_norm_type='L2-Hys'):
        '''
        Extract HOG Features for the given image.

        If the given image is multichannel, by default all channel features are extracted
        The channel arg can be set to a list to extract specific channel features
        For eg, channels = [ 0, 2] will only extract features from the 1st and 3rd channel
        '''
        num_channels = img.shape[2] if len(img.shape) > 2 else 1

        if num_channels == 1:
            return self.get_features_single_channel(img, feature_vec, vis, block_norm_type)

        # If its a multichannel image determin which channels to use
        if not isinstance(channel, list) and channel != 'ALL':
            raise ValueError('Channel must either be ALL or list of channel index to use')

        channels_to_use = []

        if isinstance(channel, list):
            channels_to_use = channel
        elif channel == 'ALL':
            channels_to_use = list(range(num_channels))

        features = []
        hog_images = []
        for channel in channels_to_use:
            output = self.get_features_single_channel(img[:, :, channel], feature_vec, vis,
                                                      block_norm_type)
            if vis:
                features.append(output[0])
                hog_images.append(output[1])
            else:
                features.append(output)
        if vis:  # Return the visualize images along with the feature arrays
            return features, hog_images

        return features
