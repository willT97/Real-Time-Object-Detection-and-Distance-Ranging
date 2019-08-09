import cv2
import os

###############################################################################
# settings for datsets in general

# master_path_to_dataset = "/tmp/pedestrian"
master_path_to_dataset = ""

# data location - training examples
DATA_training_path_neg = os.path.join(master_path_to_dataset,
                                      "../INRIAPerson/Train/neg/")
DATA_training_path_pos = os.path.join(master_path_to_dataset,
                                      "../INRIAPerson/train_64x128_H96/pos/")

# data location - testing examples
DATA_testing_path_neg = os.path.join(master_path_to_dataset,
                                     "../INRIAPerson/Test/neg/")
DATA_testing_path_pos = os.path.join(master_path_to_dataset,
                                     "..//INRIAPerson/test_64x128_H96/pos/")

# vehicle data location - training examples
VEHICLE_DATA_training_path_neg = os.path.join(
    master_path_to_dataset,
    "/media/mxvg25/Datasets/computer_vision/non-vehicles/GTI")
# maybe train on extras aswell after
VEHICLE_DATA_training_path_pos = os.path.join(
    master_path_to_dataset,
    "/home/hudson/ug/mxvg25/Documents/computer_vision/vehicles/KITTI_extracted"
)

# size of the sliding window patch / image patch to be used for classification
# (for larger windows sizes, for example from selective search - resize the
# window to this size before feature descriptor extraction / classification)

DATA_WINDOW_SIZE = [64, 128]

# the maximum left/right, up/down offset to use when generating samples
# for training
# that are centred around the centre of the image

DATA_WINDOW_OFFSET_FOR_TRAINING_SAMPLES = 3

DATA_training_sample_count_neg = 5
DATA_training_sample_count_pos = 10
VEHICLE_training_sample_count_pos = 0

# class names - N.B. ordering of 0, 1 for neg/pos = order of paths

DATA_CLASS_NAMES = {"other": 0, "pedestrian": 1, "vehicle": 2}

###############################################################################
# settings for HOG approaches

# C and gamma values
HOG_C_VALUE = 0.5
HOG_GAMMA_VALUE = 0.0078125

HOG_SVM_MULTI_PATH = "VEHICLE_svm_hog.xml"
HOG_SVM_PED_PATH = "svm_hog.xml"

HOG_SVM_kernel = cv2.ml.SVM_RBF  # see opencv manual for other options
HOG_SVM_max_training_iterations = 600  # stop training after max iterations

###############################################################################
# Selective Search Parameters

camera_focal_length_px = 399.9745178222656  # focal length in pixels
camera_focal_length_m = 4.8 / 1000  # focal length in metres (4.8 mm)
stereo_camera_baseline_m = 0.2090607502  # camera baseline in metres

image_centre_h = 262.0
image_centre_w = 474.5
# Z = ((f*B) / 2)

###############################################################################
