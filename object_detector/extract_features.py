import argparse as ap
import os

import scipy.misc
from skimage.feature import hog
from sklearn.externals import joblib
from tqdm import tqdm

from config import *
from utils import rgb2gray


def extract_features(image_dir_path, feature_dir_path, n_samples, ext='.feat'):
    progress_bar = tqdm(total=n_samples)
    i = 0
    for image_path in os.listdir(image_dir_path):
        if i == n_samples:
            break

        image = scipy.misc.imread(os.path.join(image_dir_path, image_path))
        image = rgb2gray(image)

        features = hog(image, orientations=ORIENTATIONS, pixels_per_cell=PIXELS_PER_CELL,
                       cells_per_block=CELLS_PER_BLOCK, visualise=VISUALISE, normalise=NORMALISE)

        features_file_name = image_path.split('.')[0] + ext
        features_dir_path = feature_dir_path
        features_file_path = os.path.join(features_dir_path, features_file_name)
        joblib.dump(features, features_file_path, compress=3)

        i += 1
        progress_bar.update(1)


if __name__ == '__main__':
    # Argument Parser
    parser = ap.ArgumentParser()
    parser.add_argument('-pi', '--pos_image_dir_path', help='Path to pos images',
                        required=True)
    parser.add_argument('-ni', '--neg_image_dir_path', help='Path to neg images',
                        required=True)
    parser.add_argument('-pf', '--pos_features_path', help='Path to the positive features directory',
                        required=True)
    parser.add_argument('-nf', '--neg_features_path', help='Path to the negative features directory',
                        required=True)
    args = vars(parser.parse_args())

    pos_image_dir_path = args['pos_image_dir_path']
    neg_image_dir_path = args['neg_image_dir_path']
    pos_features_path = args['pos_features_path']
    neg_features_path = args['neg_features_path']

    # If feature directories don't exist, create them
    if not os.path.exists(pos_features_path):
        os.makedirs(pos_features_path)

    # If feature directories don't exist, create them
    if not os.path.exists(neg_features_path):
        os.makedirs(neg_features_path)

    print('Calculating descriptors for the training samples and saving them')

    print('Positive samples extracting ...')
    extract_features(image_dir_path=pos_image_dir_path, feature_dir_path=pos_features_path, n_samples=POS_SAMPLES)

    print('Negative samples extracting ...')
    extract_features(image_dir_path=neg_image_dir_path, feature_dir_path=neg_features_path, n_samples=NEG_SAMPLES)

    print('Completed calculating features from training images')
