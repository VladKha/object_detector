import os

data_path = '../data'
datasets_path = data_path + '/datasets'

# Links to datasets
# WIDERFace http://mmlab.ie.cuhk.edu.hk/projects/WIDERFace/
WIDER_train_dataset_url = 'https://doc-10-24-docs.googleusercontent.com/docs/securesc/ha0ro937gcuc7l7deffksulhg5h7mbp1/2s78i51f2267nrg01r8cv4g59reau0nj/1512813600000/13356917703944334611/*/0B6eKvaijfFUDQUUwd21EckhUbWs?e=download'
WIDER_train_dataset_path = datasets_path + '/WIDER/WIDER_train.zip'
WIDER_train_face_annotations_url = 'http://mmlab.ie.cuhk.edu.hk/projects/WIDERFace/support/bbx_annotation/wider_face_split.zip'
WIDER_train_face_annotations_path = datasets_path + '/WIDER/wider_face_split.zip'
# CelebA http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html
CelebA_train_dataset_url = 'https://doc-0s-84-docs.googleusercontent.com/docs/securesc/ha0ro937gcuc7l7deffksulhg5h7mbp1/d4gfinnv6t4r3bmdlg598c1o0e6uoe70/1512813600000/13182073909007362810/*/0B7EVK8r0v71pZjFTYXZWM3FlRnM?e=download'
CelebA_train_dataset_path = datasets_path + '/CelebA/img_align_celeba.zip'

# create dirs for datasets
if not os.path.exists(datasets_path + '/WIDER'):
    os.mkdir(datasets_path + '/WIDER')
if not os.path.exists(datasets_path + '/CelebA'):
    os.mkdir(datasets_path + '/CelebA')

# Fetch and extract datasets
for (url, path) in [
    (WIDER_train_dataset_url, WIDER_train_dataset_path),
    (WIDER_train_face_annotations_url, WIDER_train_face_annotations_path),
    (CelebA_train_dataset_url, CelebA_train_dataset_path)
]:
    if not os.path.exists(path):
        os.system('wget {} -O {}'.format(url, path))
        os.system('unzip -d {} {}'.format(os.path.dirname(path), path))

images_dir_path = datasets_path + '/WIDER/WIDER_train/images'
face_annotations_file = datasets_path + '/WIDER/wider_face_split/wider_face_train_bbx_gt.txt'

pos_images_path = datasets_path + '/CelebA/img_align_celeba'
neg_images_path = datasets_path + '/WIDER/WIDER_neg'

# create negative samples
os.system('python3 ../object_detector/create_neg_samples_WIDER.py -i {} -lf {} -n {}'.format(
    images_dir_path, face_annotations_file, neg_images_path
))

# Extract the features
pos_features_path = data_path + '/features/pos'
neg_features_path = data_path + '/features/neg'
os.system('python3 ../object_detector/extract_features.py -pi {} -ni {} -pf {} -nf {}'.format(
    pos_images_path, neg_images_path, pos_features_path, neg_features_path
))

# Perform training
os.system('python3 ../object_detector/train_classifier.py -p {} -n {}'.format(
    pos_features_path, neg_features_path
))

# Perform testing
test_images_path = datasets_path + '/test_images'
os.system(
    'python3 ../object_detector/test_classifier.py -i {} --visualize'.format(test_images_path)
)
