#!/usr/bin/env python3

import cv2
import numpy as np
import time, os
from threading import Thread
import select, socket, sys, struct
import logging

THRESH = 110
try:
    import cPickle as pickle
except:
    import pickle


from multiprocessing.connection import Listener

### Initialize ###

cv2.namedWindow("cam", cv2.WINDOW_OPENGL+cv2.WINDOW_AUTOSIZE)
cap = cv2.VideoCapture(0)
cap.set(3,1920)
robot_positions = {}
SERVER_ADDR = ("255.255.255.255", 50008)
RECV_BUFFER = 128  # Block size
MIN_CONTOUR_AREA = 100
MAX_CONTOUR_AREA = 10000
ROI_IMAGE_SIZE = 30
logging.basicConfig(#filename='position_server.log',     # To a file. Or not.
                    filemode='w',                       # Start each run with a fresh log
                    format='%(asctime)s, %(levelname)s, %(message)s',
                    datefmt='%H:%M:%S',
                    level=logging.INFO, )              # Log info, and warning
running = True
n = 100             # Number of loops to wait for time calculation
t = time.time()

try:
    npaClassifications = np.loadtxt("classifications.txt", np.float32)  # read in training classifications
except:
    print("error, unable to open classifications.txt, exiting program\n")

try:
    npaFlattenedImages = np.loadtxt("flattened_images.txt", np.float32)  # read in training images
except:
    print("error, unable to open flattened_images.txt, exiting program\n")

npaClassifications = npaClassifications.reshape(
    (npaClassifications.size, 1))  # reshape numpy array to 1d, necessary to pass to call to train

kNearest = cv2.ml.KNearest_create()  # instantiate KNN object

kNearest.train(npaFlattenedImages, cv2.ml.ROW_SAMPLE, npaClassifications)

### Helper functions ###
def atan2_vec(vector):
    return -np.arctan2(vector[1], vector[0])


def vec_length(vector):
    return np.dot(vector, vector)**0.5


def pixel(img_grey, vector):
    if img_grey[vector[1], vector[0]]:
        return 1
    else:
        return 0

def rotate(image, angle, center=None, scale=1.0):
    (h, w) = image.shape[:2]

    if center is None:
        center = (w / 2, h / 2)

    # Perform the rotation
    M = cv2.getRotationMatrix2D(center, angle, scale)
    rotated = cv2.warpAffine(image, M, (w, h))

    return rotated

def crop(img, target_w, target_h):
    w, h = img.shape[:2]
    y1 = max(0, w // 2 - target_w // 2)
    y2 = min(w, w // 2 + target_w // 2)
    x1 = max(0, h // 2 - target_h // 2)
    x2 = min(h, h // 2 + target_h // 2)
    return np.array(img[y1:y2, x1:x2])

def crop_minAreaRect(img, rect, extra_crop = 0):

    # rotate img
    (h, w) = img.shape[:2]
    center = rect[0]
    angle = rect[2]
    scale = 1.0
    print(center, angle, scale)

    # Perform the rotation
    M = cv2.getRotationMatrix2D(center, angle, scale)
    img_rot = cv2.warpAffine(img, M, (w, h))

    # rotate bounding box
    print(rect)
    box = cv2.boxPoints(rect)
    pts = cv2.transform(np.array([box]), M)[0].astype(int)
    pts[pts < 0] = 0

    # crop
    img_crop = img_rot[pts[1][1]+extra_crop:pts[0][1]-extra_crop,
                       pts[1][0]+extra_crop:pts[2][0]-extra_crop]

    return np.array(img_crop)


### Thread(s) ###

class SocketThread(Thread):
    def __init__(self):
        # Initialize server socket
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        logging.info("Position server started on UDP {0}".format(SERVER_ADDR))
        Thread.__init__(self)

    def run(self):
        global robot_positions, running

        while running:
            data = pickle.dumps(robot_positions)

            try:
                sent = self.server_socket.sendto(data, SERVER_ADDR)
                # print(sent)
                time.sleep(0.025)
            except OSError as exc:
                if exc.errno == 55:
                    time.sleep(0.1)
                else:
                    raise
        self.server_socket.close()
        logging.info("Socket server stopped")



### Start it all up ###
# socket_server = SocketThread()
# socket_server.start()

while True:


    ok, img = cap.read()
    if not ok:
        continue    # and try again.

    img = crop(img, 400, 500)
    height, width = img.shape[:2]

    # print(img_crop.shape)

    # convert to grayscale
    img_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # logging.debug("read greyscale", t - time.time())
    # Otsu's thresholding. Nice & fast!
    # http://docs.opencv.org/trunk/d7/d4d/tutorial_py_thresholding.html
    # values, img_grey = cv2.threshold(img_grey, 200, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Simple adaptive mean thresholding
    values, img_grey = cv2.threshold(img_grey, 50, 255, cv2.ADAPTIVE_THRESH_MEAN_C)

    # Find contours and tree
    img_grey, contours, hierarchy = cv2.findContours(img_grey, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # logging.debug("found contours", t - time.time())

    # Preview thresholded image
    cv2.cvtColor(img_grey, cv2.COLOR_GRAY2BGR)

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

            # Cut the minimum rectangle from the image
            die = crop_minAreaRect(img_grey, rotated_rect, extra_crop = 5)
            cv2.imshow("crop", die)

            imgROIResized = cv2.resize(die, (ROI_IMAGE_SIZE,
                                                ROI_IMAGE_SIZE))  # resize image, this will be more consistent for recognition and storage

            npaROIResized = imgROIResized.reshape((1, ROI_IMAGE_SIZE ** 2))
            npaROIResized = np.float32(npaROIResized)  # convert from 1d numpy array of ints to 1d numpy array of floats

            retval, npaResults, neigh_resp, dists = kNearest.findNearest(npaROIResized,
                                                                         k=1)  # call KNN function find_nearest
            print(npaResults, neigh_resp, dists)
            code = str(chr(int(npaResults[0][0])))

            cv2.putText(img,
                            u"code: {0}".format(code),
                            (int(rotated_rect[0][0]),int(rotated_rect[0][1])),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)






    # Show all calculations in the preview window
    cv2.imshow("cam", img)

    intValidChars = [ord('0'), ord('1'), ord('2'), ord('3'), ord('4'), ord('5'), ord('6'), ord('7'), ord('8'), ord('9')]

    # Wait for the 'q' key. Dont use ctrl-c !!!
    keypress = cv2.waitKey(3000) & 0xFF
    if keypress == ord('q'):
        np.savetxt("classifications.txt", npaClassifications)
        np.savetxt("flattened_images.txt", npaFlattenedImages)
        break
    elif keypress in intValidChars:  # else if the char is in the list of chars we are looking for . . .
        # Add classifier
        new_classifier = np.array([[keypress]]).astype(np.float32)
        npaClassifications = np.append(npaClassifications, new_classifier, axis=0)  # append classification char to integer list of chars (we will convert to float later before writing to file)
        npaFlattenedImages = np.append(npaFlattenedImages, npaROIResized.astype(np.float32), 0)
        # Retrain
        kNearest = cv2.ml.KNearest_create()
        kNearest.train(npaFlattenedImages, cv2.ml.ROW_SAMPLE, npaClassifications)



        ### clean up ###
running = False
cap.release()
cv2.destroyAllWindows()
logging.info("Cleaned up")