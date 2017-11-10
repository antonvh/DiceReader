#!/usr/bin/env python3

import cv2
import numpy as np
import time, os
from threading import Thread
import select, socket, sys, struct
import logging
from firebase import firebase

### Initialize ###
def nothing(x): # An empty handler for openCV sliders
    pass

# Yellow filter
min_H = 15
min_S = 140
min_V = 115
max_H = 24
max_S = 255
max_V = 255
thresh = 50
v_crop_offset = 25
h_crop_offset = 67

# Other Image Recog settings
MIN_CONTOUR_AREA = 10000
MAX_CONTOUR_AREA = 40000
ROI_IMAGE_SIZE = 40


# Create window & trackbars
cv2.namedWindow("cam", cv2.WINDOW_OPENGL)
cv2.createTrackbar('min_H', 'cam', 0, 255, nothing)
cv2.createTrackbar('min_S', 'cam', 0, 255, nothing)
cv2.createTrackbar('min_V', 'cam', 0, 255, nothing)
cv2.createTrackbar('max_H', 'cam', 0, 255, nothing)
cv2.createTrackbar('max_S', 'cam', 0, 255, nothing)
cv2.createTrackbar('max_V', 'cam', 0, 255, nothing)
cv2.createTrackbar('thresh', 'cam', 0, 255, nothing)
cv2.createTrackbar('h_crop_offset', 'cam', 0, 255, nothing)
cv2.createTrackbar('v_crop_offset', 'cam', 0, 255, nothing)
cv2.setTrackbarPos('min_H', 'cam', min_H)
cv2.setTrackbarPos('min_S', 'cam', min_S)
cv2.setTrackbarPos('min_V', 'cam', min_V)
cv2.setTrackbarPos('max_H', 'cam', max_H)
cv2.setTrackbarPos('max_S', 'cam', max_S)
cv2.setTrackbarPos('max_V', 'cam', max_V)
cv2.setTrackbarPos('thresh', 'cam', thresh)
cv2.setTrackbarPos('h_crop_offset', 'cam', h_crop_offset)
cv2.setTrackbarPos('v_crop_offset', 'cam', v_crop_offset)

# Start webcam
cap = cv2.VideoCapture(0)
cap.set(3, 1920)
cap.set(4, 1080)

# IO
firebase = firebase.FirebaseApplication('https://dungeonsanddragons-6a9b1.firebaseio.com/', authentication=None)

### Load recognition data ###
try:
    npaClassifications = np.loadtxt("classifications.txt", np.float32)  # read in training classifications
except:
    print("error, unable to open classifications.txt, creating empty array")
    npaClassifications = np.array([0], np.float32)

try:
    npaFlattenedImages = np.loadtxt("flattened_images.txt", np.float32)  # read in training images
except:
    print("error, unable to open flattened_images.txt, creating empty array")
    npaFlattenedImages = np.ones((1, ROI_IMAGE_SIZE**2), np.float32)

npaClassifications = npaClassifications.reshape(
    (npaClassifications.size, 1))  # reshape numpy array to 1d, necessary to pass to call to train

kNearest = cv2.ml.KNearest_create()  # instantiate KNN object

kNearest.train(npaFlattenedImages, cv2.ml.ROW_SAMPLE, npaClassifications)

### Helper functions ###
def rotate(image, angle, center=None, scale=1.0):
    (h, w) = image.shape[:2]

    if center is None:
        center = (w / 2, h / 2)

    # Perform the rotation
    M = cv2.getRotationMatrix2D(center, angle, scale)
    rotated = cv2.warpAffine(image, M, (w, h))

    return rotated

def crop(img, target_w, target_h, center=None):
    h, w = img.shape[:2]
    if center == None: center = (w // 2, h // 2)
    x1 = max(0, center[0] - target_w // 2)
    x2 = min(w, center[0] + target_w // 2)
    y1 = max(0, center[1] - target_h // 2)
    y2 = min(h, center[1] + target_h // 2)
    return np.array(img[y1:y2, x1:x2])

def crop_minAreaRect(img, rect, extra_crop = 0):

    # rotate img
    (h, w) = img.shape[:2]
    center = rect[0]
    angle = rect[2]
    scale = 1.0

    # Perform the rotation
    M = cv2.getRotationMatrix2D(center, angle, scale)
    img_rot = cv2.warpAffine(img, M, (w, h))

    # rotate bounding box
    box = cv2.boxPoints(rect)
    pts = cv2.transform(np.array([box]), M)[0].astype(int)
    pts[pts < 0] = 0

    # crop
    top = max(0, pts[1][0]+extra_crop)
    left = max(0, pts[1][1]+extra_crop)
    bottom = min(h, pts[2][0]-extra_crop)
    right = min(w, pts[0][1]-extra_crop)
    if right <= left : right = left+extra_crop
    if bottom <= top : bottom = top+extra_crop
    img_crop = img_rot[left:right,
                       top:bottom]

    return np.array(img_crop)


### Main recognition loop ###
while True:
    ok, img = cap.read()
    if not ok:
        continue    # and try again.

    # img = crop(img, 500, 500)
    height, width = img.shape[:2]

    # Filter yellow
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    lower_yellow = (min_H,min_S, min_V)
    upper_yellow = (max_H, max_S, max_V)

    mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
    mask = cv2.bitwise_not(mask)

    # convert to grayscale
    img_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # logging.debug("read greyscale", t - time.time())
    # Otsu's thresholding. Nice & fast!
    # http://docs.opencv.org/trunk/d7/d4d/tutorial_py_thresholding.html
    # values, img_grey = cv2.threshold(img_grey, thresh, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Simple adaptive mean thresholding
    # values, img_grey = cv2.threshold(img_grey, thresh, 255, cv2.ADAPTIVE_THRESH_MEAN_C)

    # Find contours and tree
    img_grey, contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Preview thresholded image
    # img = cv2.cvtColor(img_grey, cv2.COLOR_GRAY2BGR)

    for x in range(0, len(contours)):
        contour = contours[x]
        area = cv2.contourArea(contour)
        k = x
        c = 0

        # Look for children with exactly one parent

        while (hierarchy[0][k][3] != -1):
            # As long as k has a first_child [2], find that child and look for children again.
            # http://docs.opencv.org/3.1.0/d9/d8b/tutorial_py_contours_hierarchy.html
            k = hierarchy[0][k][3]
            c = c + 1

        if hierarchy[0][k][3] != -1:
            c = c + 1

        if c == 1 and  MIN_CONTOUR_AREA < area < MAX_CONTOUR_AREA:
            # Fit a rectangle around the found contour and draw it.
            rotated_rect = cv2.minAreaRect(contour) #(center, size, angle)
            rotation = rotated_rect[2]
            box = cv2.boxPoints(rotated_rect).astype(int)
            cv2.drawContours(img, [box], -1, (0, 255, 0))

            # Fit a circle and draw it
            circle = cv2.minEnclosingCircle(contour)  # (center, size, angle)
            radius = int(circle[1])
            center = tuple(np.array(circle[0], int))
            offset_center_x = center[0] + int((center[0] - width / 2)/ width * h_crop_offset)
            offset_center_y = center[1] + int((center[1] - height)/ height * v_crop_offset)
            # print()
            cv2.circle(img, center, radius, (0, 255, 0))
            cv2.circle(img, (offset_center_x, offset_center_y), radius//2, (0, 255, 0))

            # Cut the minimum rectangle from the image
            # die = crop_minAreaRect(img_grey, rotated_rect, extra_crop = 20)
            die = crop(img_grey, radius-10, radius-10, center=(offset_center_x, offset_center_y))
            try:
                cv2.imshow("crop", die)
            except:
                print(radius, offset_center_x, offset_center_y)
                raise

            # Resize the result to the size for recognition and find nearest.
            imgROIResized = cv2.resize(die, (ROI_IMAGE_SIZE, ROI_IMAGE_SIZE))
            npaROIResized = imgROIResized.reshape((1, ROI_IMAGE_SIZE ** 2))
            npaROIResized = np.float32(npaROIResized)  # convert from 1d numpy array of ints to 1d numpy array of floats
            retval, npaResults, neigh_resp, dists = kNearest.findNearest(npaROIResized, k=1)
            die_roll = int(npaResults[0][0])

            # Write the result in firebase
            result = firebase.get('/dice', None)
            if result:
                keys = list(result.keys())
                if keys is not None and len(keys) > 0:
                    firebase.delete('/dice', keys[0])
            firebase.post('/dice', {'d20': [die_roll], 'd6': [die_roll]})

            # Draw the result on screen
            cv2.putText(img,
                            u"code: {0}".format(str(die_roll)),
                        (int(rotated_rect[0][0]),int(rotated_rect[0][1])),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 255), 2)

    # Show all calculations in the preview window
    cv2.imshow("cam", img)

    # Keys for training. The second row is 11, 12, 13 etc...
    intValidChars = [ord('1'), ord('2'), ord('3'), ord('4'), ord('5'), ord('6'), ord('7'), ord('8'), ord('9'), ord('0'),
                     ord('q'), ord('w'), ord('e'), ord('r'), ord('t'), ord('y'), ord('u'), ord('i'), ord('o'), ord('p')]

    # Wait for the 'q' key. Dont use ctrl-c !!!
    keypress = cv2.waitKey(1) & 0xFF

    # Read trackbar values
    min_H = cv2.getTrackbarPos('min_H', 'cam')
    min_S = cv2.getTrackbarPos('min_S', 'cam')
    min_V = cv2.getTrackbarPos('min_V', 'cam')
    max_H = cv2.getTrackbarPos('max_H', 'cam')
    max_S = cv2.getTrackbarPos('max_S', 'cam')
    max_V = cv2.getTrackbarPos('max_V', 'cam')
    thresh = cv2.getTrackbarPos('thresh', 'cam')
    h_crop_offset = cv2.getTrackbarPos('h_crop_offset', 'cam')
    v_crop_offset = cv2.getTrackbarPos('v_crop_offset', 'cam')

    if keypress == ord('x'):
        np.savetxt("classifications.txt", npaClassifications)
        np.savetxt("flattened_images.txt", npaFlattenedImages)
        break
    elif keypress in intValidChars:  # else if the char is in the list of chars we are looking for . . .
        # Add classifier in 4 rotations
        for i in range(4):
            new_classifier = np.array([[intValidChars.index(keypress)+1]]).astype(np.float32)
            npaClassifications = np.append(npaClassifications, new_classifier, axis=0)  # append classification char to integer list of chars (we will convert to float later before writing to file)
            cv2.rotate(npaROIResized, cv2.ROTATE_90_CLOCKWISE)
            npaFlattenedImages = np.append(npaFlattenedImages, npaROIResized.astype(np.float32), 0)
        # Retrain the network
        kNearest = cv2.ml.KNearest_create()
        kNearest.train(npaFlattenedImages, cv2.ml.ROW_SAMPLE, npaClassifications)



### clean up ###
cap.release()
cv2.destroyAllWindows()