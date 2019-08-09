import cv2
import os
import math
import numpy as np
import params
from utils import *
import stereo_disparity
import selective_search

# loads in the svm if it exists
# The main file that runs the object detection on the dataset

# master_path_to_dataset = "../INRIAPerson/Test/pos/" # edit this
master_path_to_dataset = "/Users/will/Documents/SSA/Computer_vision/TTBB-durham-02-10-17-sub10"


def main():
    try:
        svm = cv2.ml.SVM_load(params.HOG_SVM_PED_PATH)
    except:
        print("Missing files - SVM!")
        print("-- have you performed training to produce these files ?")
        exit()

    ##########################################################################

    # press all the go-faster buttons - i.e. speed-up using multithreads

    cv2.setUseOptimized(True)
    cv2.setNumThreads(4)

    ##########################################################################

    # directory_to_cycle_left = "left-images"
    full_path_directory_left = os.path.join(master_path_to_dataset,
                                            "left-images")

    ##########################################################################
    # create Selective Search Segmentation Object using default parameters

    ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()

    ##########################################################################

    for filename in sorted(os.listdir(full_path_directory_left)):
        # starts a counter
        start_t = cv2.getTickCount()

        full_path_filename_left, full_path_filename_right = stereo_disparity.get_lr_pair(
            filename)

        # checks to see if the input is an image and that there is a
        # corresponding right image to the current left image
        if ('.png' in filename) and (os.path.isfile(full_path_filename_right)):

            # retrieves the disparity from the filename of left image
            # to find the region boxes in selective search
            disparity = stereo_disparity.get_disparity_from_image(filename)

            # retrieves disparity frame used for calculating distance
            distance_disparity = stereo_disparity.get_distance_disparity(
                filename)

            # normalize so can view the image for selective search map
            disp_frame = cv2.normalize(
                src=disparity,
                dst=None,
                beta=0,
                alpha=255,
                norm_type=cv2.NORM_MINMAX)
            disp_frame = np.uint8(disp_frame)
            disp_frame = stereo_disparity.gamma_correction(disp_frame, 0.7)

            # read in the left image
            img_frame = cv2.imread(
                os.path.join(full_path_directory_left, filename),
                cv2.IMREAD_COLOR)

            # convert disp_frame to a 3 channel image
            disp_frame = np.stack((disp_frame, ) * 3, axis=-1)

            # return all of the rectangles from the image
            rects = selective_search.getRects_frame(disp_frame, 1000, ss)

            selective_search.displace_rects(rects, 135)

            # clear the rectangles from the image and display only the detections

            img_frame = cv2.imread(
                os.path.join(full_path_directory_left, filename),
                cv2.IMREAD_COLOR)

            # returns the positive regions and the rejected search regions
            # also draws the bounding boxes around the detections
            pos_rects, neg_rects, z_values = selective_search.get_result(
                rects, img_frame, svm, distance_disparity, disp_frame)

            if len(z_values) == 0:
                z_min = 0.0
            else:
                z_min = min(z_values)

            print(filename)
            print('{0} : nearest detected scene object ({1:.1f}m)'.format(filename.replace("_L", "_R"), z_min))

            cv2.imshow('Selective Search - Positive Object detections',
                       img_frame)

            ###################################################################

            stop_t = (
                (cv2.getTickCount() - start_t) / cv2.getTickFrequency()) * 1000

            #print('Processing time (ms): {}'.format(stop_t))

            key = cv2.waitKey(max(40, 40 - int(math.ceil(stop_t)))) & 0xFF
            # start the event loop - essential

            # cv2.waitKey() is a keyboard binding function (argument is the time in milliseconds).
            # It waits for specified milliseconds for any keyboard event.
            # If you press any key in that time, the program continues.
            # If 0 is passed, it waits indefinitely for a key stroke.
            # (bitwise and with 0xFF to extract least significant byte of multi-byte response)
            # here we use a wait time in ms. that takes account of processing time already used in the loop

            # wait 40ms or less depending on processing time taken (i.e. 1000ms / 25 fps = 40 ms)

            # e.g. if user presses "x" then exit / press "f" for fullscreen

            if (key == ord('x')):
                break
            elif (key == ord('f')):
                cv2.setWindowProperty(windowName, cv2.WND_PROP_FULLSCREEN,
                                      cv2.WINDOW_FULLSCREEN)

            ss.clearImages()

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
