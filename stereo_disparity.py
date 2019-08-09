import cv2
import os
import numpy as np
from main import master_path_to_dataset


# on linux computer
#master_path_to_dataset = "/media/mxvg25/Datasets/computer_vision/TTBB-durham-02-10-17-sub10"
directory_to_cycle_left = "left-images"
directory_to_cycle_right = "right-images"

# set this to a file timestamp to start from
# e.g. set to 1506943191.487683 for the end of the Bailey
# just as the vehicle turns

skip_forward_file_pattern = ""
# set to timestamp to skip forward to

crop_disparity = True
# display full or cropped disparity image
pause_playback = False
# pause until key press after each image

#####################################################################

# resolve full directory location of data set for left / right images

full_path_directory_left = os.path.join(master_path_to_dataset,
                                        directory_to_cycle_left)
full_path_directory_right = os.path.join(master_path_to_dataset,
                                         directory_to_cycle_right)

# get a list of the left image files and sort them (by timestamp in filename)
left_file_list = sorted(os.listdir(full_path_directory_left))

max_disparity = 128

# for the distance
distanceProcessor = cv2.StereoSGBM_create(0, max_disparity, 21)

# stereoProcessor = cv2.StereoSGBM_create(-64, 192, 5, 600, 2400, 10, 4, 1, 150, 2)

# For the selective search

stereoProcessor = cv2.StereoSGBM_create(
    0,
    max_disparity,
    5,
    600,
    2400,
    1,
    63,
    15,
    0,
    2,
    mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY)

right_matcher = cv2.ximgproc.createRightMatcher(stereoProcessor)

# applying disparity map post-filting
# from the open cv docs
# https://docs.opencv.org/3.4.2/d3/d14/tutorial_ximgproc_disparity_filtering.html
# creating a wls filter
# FILTER Parameters
lmbda = 80000
sigma = 1.2
visual_multiplier = 1.0

wls_filter = cv2.ximgproc.createDisparityWLSFilter(
    matcher_left=stereoProcessor)
wls_filter.setLambda(lmbda)
wls_filter.setSigmaColor(sigma)


###############################################################################

# gets the filepaths for left and right files
def get_lr_pair(filename_left):

    # from the left image get the right image filename
    filename_right = filename_left.replace("_L", "_R")

    # prints out the filepaths for left and right to check they are correct
    full_path_filename_left = os.path.join(full_path_directory_left,
                                           filename_left)
    full_path_filename_right = os.path.join(full_path_directory_right,
                                            filename_right)

    return full_path_filename_left, full_path_filename_right


# checks the files exists and returns the cv2 images
def get_lr_images(filename_left, full_path_filename_left,
                  full_path_filename_right):
    if ('.png' in filename_left) and (
            os.path.isfile(full_path_filename_right)):

        # read left and right images and display in windows
        # N.B. despite one being grayscale both are in fact stored as 3-channel
        # RGB images so load both as such

        imgL = cv2.imread(full_path_filename_left, cv2.IMREAD_COLOR)


        imgR = cv2.imread(full_path_filename_right, cv2.IMREAD_COLOR)

        return imgL, imgR
    else:
        return "", ""


###############################################################################

def lr_preprocessing(imgL, imgR):
    # remember to convert to grayscale (as the disparity matching works on grayscale)
    # N.B. need to do for both as both are 3-channel images

    grayL = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)
    grayR = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)

    # applying CLAHE to left and right images
    clahe = cv2.createCLAHE(clipLimit=32.0, tileGridSize=(8, 8))
    grayL = clahe.apply(grayL)
    grayR = clahe.apply(grayR)

    # apply gaussian blur to smoothen the image
    grayL = cv2.GaussianBlur(grayL, (5, 5), 0)
    grayR = cv2.GaussianBlur(grayR, (3, 3), 0)

    # followed by median blur to reduce noise
    grayL = cv2.medianBlur(grayL, 3)
    grayR = cv2.medianBlur(grayR, 3)

    return grayL, grayR

###############################################################################

def gamma_correction(img, correction):
    img = img / 255.0
    img = cv2.pow(img, correction)
    return np.uint8(img * 255)

###############################################################################

# returns the disparity from left and right images
def get_disparity(imgL, imgR):
    # applies histogram equalization and filters to the left and right images
    grayL, grayR = lr_preprocessing(imgL, imgR)

    # compute disparity image from undistorted and rectified stereo images
    # (which for reasons best known to the OpenCV developers is returned scaled by 16)

    displ = stereoProcessor.compute(grayL, grayR)  # .astype(np.float32)/16
    dispr = right_matcher.compute(grayR, grayL)  # .astype(np.float32)/16
    displ = np.int16(displ)
    dispr = np.int16(dispr)
    filteredImg = wls_filter.filter(displ, grayL, None, dispr)

    return filteredImg


#  gets the cropped disparity without the left side of the image
#  and the car bonnet
def get_cropped_disparity(disparity):
    # converts to single channel image
    width = np.size(disparity, 1)

    # 135 is the amount needed to add on to the boxes found from selective
    # search of the images
    disparity = disparity[0:390, 135:width]
    return disparity


# returns the cropped disparity from the left filename image
def get_disparity_from_image(filename_left):
    full_path_filename_left, full_path_filename_right = get_lr_pair(
        filename_left)
    imgL, imgR = get_lr_images(filename_left, full_path_filename_left,
                               full_path_filename_right)
    disparity = get_disparity(imgL, imgR)

    cropped_disparity = get_cropped_disparity(disparity)
    return cropped_disparity


# gets the disparity used for the distance not the one used for selective search
def get_distance_disparity(filename_left):
    # gets the left and right filenames
    full_path_filename_left, full_path_filename_right = get_lr_pair(
        filename_left)

    # returns the images from the filenames
    imgL, imgR = get_lr_images(filename_left, full_path_filename_left,
                               full_path_filename_right)
    # adds preprocessing to the images
    grayL, grayR = lr_preprocessing(imgL, imgR)
    disparity = distanceProcessor.compute(grayL, grayR)

    dispNoiseFilter = 5
    # increase for more agressive filtering
    # removes disparity noise
    cv2.filterSpeckles(disparity, 0, 4000, max_disparity - dispNoiseFilter)

    # scale the disparity to 8-bit for viewing
    # divide by 16 and convert to 8-bit image (then range of values should
    # be 0 -> max_disparity) but in fact is (-1 -> max_disparity - 1)
    # so we fix this also using a initial threshold between 0 and max_disparity
    # as disparity=-1 means no disparity available

    _, disparity = cv2.threshold(disparity, 0, max_disparity * 16,
                                 cv2.THRESH_TOZERO)
    disparity_scaled = (disparity / 16.).astype(np.uint8)

    width = np.size(disparity_scaled, 1)
    disparity_scaled = disparity_scaled[0:390, 135:width]

    return disparity_scaled

###############################################################################
