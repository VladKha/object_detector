import os

import cv2
import numpy as np


IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.bmp', 'pgm')


def rgb2gray(rgb_image):
    """
    Convert RGB image to grayscale

    Parameters:
        rgb_image : RGB image

    Returns:
        gray : grayscale image

    """
    return np.dot(rgb_image[..., :3], [0.299, 0.587, 0.144])


def pyramid(image, downscale=1.5, min_size=(30, 30)):
    # yield the original image
    yield image

    # keep looping over the pyramid
    while True:
        # compute the new dimensions of the image and resize it
        w = int(image.shape[1] / downscale)
        image = resize(image, width=w)

        # if the resized image does not meet the supplied minimum
        # size, then stop constructing the pyramid
        if image.shape[0] < min_size[1] or image.shape[1] < min_size[0]:
            break

        # yield the next image in the pyramid
        yield image


def sliding_window(image, step_size, window_size):
    """
    This function returns a patch of the input image `image` of size equal
    to `window_size`. The first image returned top-left co-ordinates (0, 0)
    and are increment in both x and y directions by the `step_size` supplied.
    :param image: input image
    :param step_size: incremented size of window
    :param window_size: size of sliding window
    :return: tuple (x, y, im_window) where
        * x is the top-left x co-ordinate
        * y is the top-left y co-ordinate
        * im_window is the sliding window image
    """
    # slide a window across the image
    for y in range(0, image.shape[0], step_size):
        for x in range(0, image.shape[1], step_size):
            # yield the current window
            yield (x, y, image[y:y + window_size[1], x:x + window_size[0]])


# Malisiewicz et al.
def non_max_suppression(boxes, overlap_thresh=0.7):
    # if there are no boxes, return an empty list
    if len(boxes) == 0:
        return []

    # if the bounding boxes integers, convert them to floats --
    # this is important since we'll be doing a bunch of divisions
    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")

    # initialize the list of picked indexes
    pick = []

    # grab the coordinates of the bounding boxes
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    # compute the area of the bounding boxes and sort the bounding
    # boxes by the bottom-right y-coordinate of the bounding box
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(y2)

    # keep looping while some indexes still remain in the indexes
    # list
    while len(idxs) > 0:
        # grab the last index in the indexes list and add the
        # index value to the list of picked indexes
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        # find the largest (x, y) coordinates for the start of
        # the bounding box and the smallest (x, y) coordinates
        # for the end of the bounding box
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])

        # compute the width and height of the bounding box
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        # compute the ratio of overlap
        overlap = (w * h) / area[idxs[:last]]

        # delete all indexes from the index list that have
        idxs = np.delete(idxs, np.concatenate(([last],
                                               np.where(overlap > overlap_thresh)[0])))

    # return only the bounding boxes that were picked using the
    # integer data type
    return boxes[pick].astype("int")


def bb_intersection(box_a, box_b):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(box_a[0], box_b[0])
    yA = max(box_a[1], box_b[1])
    xB = min(box_a[2], box_b[2])
    yB = min(box_a[3], box_b[3])

    # compute the area of intersection rectangle
    t1 = xB - xA + 1
    t2 = yB - yA + 1
    if t1 <= 0 or t2 <= 0:
        intersection_area = 0
    else:
        intersection_area = (xB - xA + 1) * (yB - yA + 1)
    return intersection_area


def bb_intersection_over_union(box_a, box_b):
    intersection_area = bb_intersection(box_a, box_b)

    # compute the area of both the prediction and ground-truth
    # rectangles
    box_a_area = (box_a[2] - box_a[0] + 1) * (box_a[3] - box_a[1] + 1)
    box_b_area = (box_b[2] - box_b[0] + 1) * (box_b[3] - box_b[1] + 1)

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the intersection area
    iou = intersection_area / (box_a_area + box_b_area - intersection_area)
    # return the intersection over union value
    return iou


def is_image_file(file_name):
    ext = file_name[file_name.rfind('.'):].lower()
    return ext in IMAGE_EXTENSIONS


def list_images(base_path, contains=None):
    # return the set of files that are valid
    return list_files(base_path, valid_exts=IMAGE_EXTENSIONS, contains=contains)


def list_files(base_path, valid_exts=IMAGE_EXTENSIONS, contains=None):
    # loop over the directory structure
    for (root_cir, dir_names, file_names) in os.walk(base_path):
        # loop over the file names in the current directory
        for file_name in file_names:
            # if the contains string is not none and the filename does not contain
            # the supplied string, then ignore the file
            if contains is not None and file_name.find(contains) == -1:
                continue

            # determine the file extension of the current file
            ext = file_name[file_name.rfind('.'):].lower()

            # check to see if the file is an image and should be processed
            if ext.endswith(valid_exts):
                # construct the path to the image and yield it
                image_path = os.path.join(root_cir, file_name).replace(" ", "\\ ")
                yield image_path


def resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # grab the image size
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
