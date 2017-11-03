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

def crop(img, target_w, target_h):
    w, h = img.shape[:2]
    y1 = max(0, w // 2 - target_w // 2)
    y2 = min(w, w // 2 + target_w // 2)
    x1 = max(0, h // 2 - target_h // 2)
    x2 = min(h, h // 2 + target_h // 2)
    return np.array(img[y1:y2, x1:x2])

### Start it all up ###
# socket_server = SocketThread()
# socket_server.start()

while True:


    ok, img = cap.read()
    if not ok:
        continue    # and try again.

    img = crop(img, 300, 400)
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
    img = cv2.cvtColor(img_grey, cv2.COLOR_GRAY2BGR)

    for x in range(0, len(contours)):
        contour = contours[x]
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

        if c == 1 and cv2.contourArea(contour) > MIN_CONTOUR_AREA:
            # Fit a rectangle around the found contour and draw it.
            rotated_rect = cv2.minAreaRect(contour)
            rotation = rotated_rect[2]
            box = cv2.boxPoints(rotated_rect).astype(int)
            cv2.drawContours(img, [box], -1, (0, 255, 0))


            dx, dy, dw, dh = cv2.boundingRect(contours[x])
            center = np.array([dx + dw // 2, dy + dh // 2])
            half_size = max(dw, dh) // 2 - 10
            half_diagonal = np.array([half_size, half_size])
            top_left = tuple(center - half_diagonal)
            bottom_right = tuple(center + half_diagonal)
            cv2.rectangle(img, top_left, bottom_right, (0,0,255))
            # Try to read the number
            imgROI = np.array(img_grey[top_left[1]: bottom_right[1], top_left[0]: bottom_right[0]])
            imgROIResized = cv2.resize(imgROI, (ROI_IMAGE_SIZE,
                                                ROI_IMAGE_SIZE))  # resize image, this will be more consistent for recognition and storage

            npaROIResized = imgROIResized.reshape((1, ROI_IMAGE_SIZE ** 2))
            npaROIResized = np.float32(npaROIResized)  # convert from 1d numpy array of ints to 1d numpy array of floats

            retval, npaResults, neigh_resp, dists = kNearest.findNearest(npaROIResized,
                                                                         k=1)  # call KNN function find_nearest
            print(npaResults, neigh_resp, dists)
            code = str(chr(int(npaResults[0][0])))

            cv2.putText(img,
                            u"code: {0}, x:{1}, y:{2}".format(code, center[0], center[1]),
                            tuple(center),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)

            # If certainty is too low, ask for confirmation and adjust learning



    # Show all calculations in the preview window
    cv2.imshow("cam", img)

    # logging.debug("shown image", t - time.time())

    # Wait for the 'q' key. Dont use ctrl-c !!!
    keypress = cv2.waitKey(2000) & 0xFF
    if keypress == ord('q'):
        break
    if n == 0:
        logging.info("Looptime: {0}, contours: {1}".format((time.time()-t)/100, len(contours)))
        n = 100
        t = time.time()
    else:
        n -= 1


### clean up ###
running = False
cap.release()
cv2.destroyAllWindows()
logging.info("Cleaned up")