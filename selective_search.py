import cv2
import os
import numpy as np
import params
from utils import *


###############################################################################

# returns the search regions for a given file
# with the number of search regions to return
def getRects_file(filename, no_of_rects):
    # if it is a PNG file
    if '.png' in filename:
        # read image from file
        frame = cv2.imread(
            os.path.join(directory_to_cycle, filename), cv2.IMREAD_COLOR)

        return getRects_frame(frame, no_of_rects), frame


# returns the rectangles from the image
def getRects_frame(frame, no_of_rects, ss):

    ss.setBaseImage(frame)

    # Switch to fast but low recall Selective Search method
    ss.switchToSelectiveSearchFast()

    # run selective search segmentation on input image
    rects = ss.process()

    # returns array of search regions defined by no_of_rects
    return rects[:no_of_rects]

###############################################################################


# draws all of the given rectangles onto the given frame
def display_ssboxes(rects, frame):
    for rect in rects:
        x, y, w, h = rect
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 1,
                      cv2.LINE_AA)


# increase the x value of rectangles by the crop size
def displace_rects(rects, crop_size):
    for rect in rects:
        rect[0] = rect[0] + crop_size

###############################################################################

# given a search region and a frame returns
# the result of svm
def get_region_prediction(rect, frame, svm):
    x, y, w, h = rect
    # cv2.imshow("frame", frame)
    # compute the hog descriptor
    rec_to_check = frame[rect[1]:rect[1] + rect[3], rect[0]:rect[0] + rect[2]]
    # resize before going into the hog descriptor and prediction
    resized = cv2.resize(rec_to_check, (64, 128), interpolation=cv2.INTER_AREA)
    img_data = ImageData(resized)

    # research

    # cv2.imshow("rec", rec_to_check)
    img_data.compute_hog_descriptor()

    # predict the value
    # print(svm.predict(np.float32([img_data.hog_descriptor])))
    retval, [result] = svm.predict(np.float32([img_data.hog_descriptor]))
    return retval, [result]


# given a region and disparity
# performs k means clustering
# with help from this tutorial
# https://www.pyimagesearch.com/2014/05/26/opencv-python-k-means-color-clustering/
# and returns the median of the
# closest cluster
def calculate_disp_val(x, y, w, h, disparity):
    # the rectangle on the disparity of the area of the rectangle on the image
    disp_box = disparity[y:y + h, (x - 135):(x - 135) + w]
    # converts to 3 channel image
    disp_box = np.stack((disp_box, ) * 3, axis=-1)

    Z = disp_box.reshape((-1, 3))
    # convert to np.float32
    Z = np.float32(disp_box)

    # define criteria, number of clusters(K) and apply kmeans()
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    K = 2
    ret, label, center = cv2.kmeans(Z, K, None, criteria, 10,
                                    cv2.KMEANS_RANDOM_CENTERS)

    center = np.uint8(center)

    # gets the center cluster array with the maximum sum e.g. the cluster with
    # the closer object
    disp = max(center, key=lambda x: x.sum())

    # median of disp center
    return int(np.median(disp))


#        Depth using Formula:
#
#               (f x B)
#      Z  =  ---------------
#            Disparity Value
#
def get_depth(disp_val):

    f = params.camera_focal_length_px
    B = params.stereo_camera_baseline_m
    Z = (f * B) / (disp_val)
    return Z

###############################################################################

# displays the positive detection bounding boxes
# returns the positive rectangles and negative rectangles
def get_result(rects, frame, svm, disparity, disp_frame):

    pos_rects, neg_rects, vehicle_rects = [], [], []

    for rect in rects:
        # retrieves prediction from proposed region
        retval, [result] = get_region_prediction(rect, frame, svm)

        # record whether there is a dectection or not
        if result[0] == params.DATA_CLASS_NAMES["pedestrian"]:
            pos_rects.append(rect)
        elif result[0] == params.DATA_CLASS_NAMES["vehicle"]:
            vehicle_rects.append(rect)
        else:
            neg_rects.append(rect)

    # remove all of the boxes where w < h for pedestrians
    pos_rects = [rect for rect in pos_rects if rect[2] < rect[3]]

    # convert rect format to calculate non max suppression
    # store as (x1,y1), (x2, y2) pairs
    pos_rects = [
        np.float32([rect[0], rect[1], rect[0] + rect[2], rect[1] + rect[3]])
        for rect in pos_rects
    ]

    # do the same for vehicle rects
    vehicle_rects = [
        np.float32([rect[0], rect[1], rect[0] + rect[2], rect[1] + rect[3]])
        for rect in vehicle_rects
    ]

    # perform non max suppression on the positive rectangles
    pos_rects = non_max_suppression_fast(np.int32(pos_rects), 0.4)

    # same for vehicle
    vehicle_rects = non_max_suppression_fast(np.int32(vehicle_rects), 0.4)
    # print("rects after non max suppression = " + str(len(pos_rects)))
    z_values = []
    final_rects = []
    for rect in pos_rects:
        # store rect as (x1, y1) (x2,y2) pairs
        x, y, x2, y2 = rect
        w = x2 - x
        h = y2 - y

        # returns the average disparity value for current region
        disp_val = calculate_disp_val(x, y, w, h, disparity)

        # if disparity value equals zero impossible to calculate distance
        if disp_val > 0:
            Z = get_depth(disp_val)
            if Z > 0 and Z < 25:
                # here is the distance
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2,
                              cv2.LINE_AA)
                z_values.append(Z)
                # add the text below the bounding box
                text = "P: {0:.2f}m".format(Z)
                cv2.putText(frame, text, (x, y + h + 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0))

                # add to the positive rects?
                # maybe change this
                final_rects.append(rect)

    for rect in vehicle_rects:
        # store rect as (x1, y1) (x2,y2) pairs
        x, y, x2, y2 = rect
        w = x2 - x
        h = y2 - y

        # returns the average disparity value for current region
        disp_val = calculate_disp_val(x, y, w, h, disparity)



        # if disparity value equals zero impossible to calculate distance
        if disp_val > 0:
            Z = get_depth(disp_val)
            if Z > 0 and Z < 25:
                # here is the distance
                # print(Z)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2,
                              cv2.LINE_AA)
                # add the text below the bounding box
                text = "V: {0:.2f}m".format(Z)
                z_values.append(Z)
                cv2.putText(frame, text, (x, y + h + 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0))

                # add to the positive rects?
                # maybe change this
                final_rects.append(rect)

    return final_rects, neg_rects, z_values


###############################################################################

# perform basic non-maximal suppression of overlapping object detections


def non_max_suppression_fast(boxes, overlapThresh):
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
        #print(overlap)

        # delete all indexes from the index list that have a significant overlap
        idxs = np.delete(
            idxs, np.concatenate(([last],
                                  np.where(overlap > overlapThresh)[0])))

    # return only the bounding boxes that were picked using the
    # integer data type
    return boxes[pick].astype("int")


################################################################################
