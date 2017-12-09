import argparse as ap
import glob
import os

from sklearn.externals import joblib
from sklearn.linear_model import SGDClassifier
from sklearn.svm import LinearSVC
from tqdm import tqdm, trange

from config import *

if __name__ == '__main__':
    # Parse the command line arguments
    parser = ap.ArgumentParser()
    parser.add_argument('-p', '--pos_features_path', help='Path to the positive features directory',
                        required=True)
    parser.add_argument('-n', '--neg_features_path', help='Path to the negative features directory',
                        required=True)
    parser.add_argument('-c', '--classifier', help='Classifier to be used', default='LIN_SVM')
    args = vars(parser.parse_args())

    pos_feat_path = args['pos_features_path']
    neg_feat_path = args['neg_features_path']

    # Classifiers supported
    clf_type = args['classifier']

    if clf_type == 'LIN_SVM':
        print('Training a Linear SVM classifier:')
        X = []
        y = []

        print('Loading positive samples...')
        progress_bar = tqdm(total=POS_SAMPLES)
        i = 0
        for feat_path in glob.glob(os.path.join(pos_feat_path, '*.feat')):
            if i == POS_SAMPLES:
                break

            x = joblib.load(feat_path)
            X.append(x)
            y.append(1)

            i += 1
            progress_bar.update(1)

        print('Loading negative samples...')
        progress_bar = tqdm(total=NEG_SAMPLES)
        i = 0
        for feat_path in glob.glob(os.path.join(neg_feat_path, '*.feat')):
            if i == NEG_SAMPLES:
                break

            x = joblib.load(feat_path)
            X.append(x)
            y.append(0)

            i += 1
            progress_bar.update(1)

        X_train = np.array(X)
        y_train = np.array(y)
        del X
        del y

        if clf_type is 'LIN_SVM':
            print('Training a Linear SVM Classifier...')
            clf = LinearSVC(random_state=RANDOM_STATE)
            clf.fit(X_train, y_train)

    elif clf_type == 'SGD':
        print('Training a SGDClassifier:')
        clf = SGDClassifier(random_state=RANDOM_STATE)

        samples = []
        print('Loading positive samples...')
        progress_bar = tqdm(total=POS_SAMPLES)
        i = 0
        for feat_path in glob.glob(os.path.join(pos_feat_path, '*.feat')):
            if i == POS_SAMPLES:
                break

            samples.append((feat_path, 1))

            i += 1
            progress_bar.update(1)

        print('Loading negative samples...')
        progress_bar = tqdm(total=NEG_SAMPLES)
        i = 0
        for feat_path in glob.glob(os.path.join(neg_feat_path, '*.feat')):
            if i == NEG_SAMPLES:
                break

            samples.append((feat_path, 0))

            i += 1
            progress_bar.update(1)

        random.shuffle(samples)

        print('Training classifier...')
        progress_bar = tqdm(total=POS_SAMPLES + NEG_SAMPLES)
        i = 0
        for i in trange(POS_SAMPLES + NEG_SAMPLES):
            feat_path, label = samples[i]
            x = joblib.load(feat_path)
            clf.partial_fit([x], [label], classes=[0, 1])

    # If model directory doesn't exist, create one
    if not os.path.isdir(os.path.split(MODEL_PATH)[0]):
        os.makedirs(os.path.split(MODEL_PATH)[0])
    joblib.dump(clf, MODEL_PATH, compress=3)
    print('Classifier saved to {}'.format(MODEL_PATH))
