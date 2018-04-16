import cv2
import numpy as np


def convert_color(img, conv='RGB2YCrCb'):  # flake8: noqa
    if conv == 'RGB2HLS':
        return cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    if conv == 'BGR2HLS':
        return cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    if conv == 'RGB2HSV':
        return cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    if conv == 'BGR2HSV':
        return cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    if conv == 'RGB2YCrCb':
        return cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    if conv == 'BGR2YCrCb':
        return cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    if conv == 'RGB2LUV':
        return cv2.cvtColor(img, cv2.COLOR_RGB2LUV)
    if conv == 'RGB2GRAY':
        return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    if conv == 'BGR2GRAY':
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if conv == 'RGB2BGR':
        return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    raise ValueError('Conversion flag for color space %s not defined' % conv)


# Define a function to draw bounding boxes
def draw_boxes(img, bboxes, color=(0, 0, 255), thick=6):
    # Make a copy of the image
    imcopy = np.copy(img)
    # Iterate through the bounding boxes
    for (startx, starty, endx, endy) in bboxes:
        # Draw a rectangle given bbox coordinates
        cv2.rectangle(imcopy, (startx, starty), (endx, endy), color, thick)
    # Return the image copy with boxes drawn
    return imcopy


# Credits to this function goes to https://github.com/jrosebr1/imutils/
# Convenience function to resize image, while maintaing the aspect ratio
def resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv2.resize(image, dim, interpolation=inter)

    # return the resized image
    return resized


def mpimg_to_cvimg(image):
    return convert_color(image, conv='RGB2BGR')


def pyramid(image, scale=1.5, min_size=(64, 64)):
    '''
    Given an image and scale, repeatedly return a scaled down 
    version of image till its smaller than the minimum size
    '''

    if scale < 1:
        raise ValueError('Scale must be lower than 1')
    # yield the original image - 1.0 Scale
    yield image

    # keep looping over the pyramid
    while True:
        # compute the new dimensions of the image and resize it
        w = int(image.shape[1] / scale)
        image = resize(image, width=w)

        # if the resized image does not meet the supplied minimum
        # size, then stop constructing the pyramid
        if image.shape[0] < min_size[1] or image.shape[1] < min_size[0]:
            break

        # yield the next image in the pyramid
        yield image


def sliding_window(image, window_size, overlap_x=0.5, overlap_y=0.5):
    step_size_x = np.int(window_size[0] * overlap_x)
    step_size_y = np.int(window_size[1] * overlap_y)
    # slide a window across the image
    for y in range(0, image.shape[0], step_size_y):
        for x in range(0, image.shape[1], step_size_x):
            # yield the current window
            yield (x, y, image[y:y + window_size[1], x:x + window_size[0]])
