# USAGE
# python3 app.py

# import the necessary packages
from collections import deque
from imutils.video import VideoStream
import numpy as np
import argparse
import cv2
import imutils
import time
import copy
import shark2Algorithm

shark2Algorithm.generate_templates()

def image_resize(image, width = None, height = None, inter = cv2.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]
    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image
    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)
    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv2.resize(image, dim, interpolation = inter)
    # return the resized image
    return resized

# region of interest (ROI) coordinates
x1, y1, x2, y2 = 100, 100, 1100, 400
# x1, y1, x2, y2 = 100, 100, 1000, 500
track_gesture_x = []
track_gesture_y = []

def find_centroid_for_each_letter(cv2, frame, a, b, letter):
    font_family = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    font_color = (255,0,0)
    font_thickness = 2
    cv2.putText(frame, letter, (a, b), font_family, font_scale, font_color, font_thickness)
    cv2.circle(frame, (a, b), 1, (0, 0, 255), -1)

def create_box_for_each_letter(cv2, frame, x1, y1, x2, y2):
    centroid_x = []
    centroid_y = []
    centroid_hash = {}
    line1 = ["Q", "W", "E", "R", "T", "Y", "U", "I", "O", "P"]
    line2 = ["A", "S", "D", "F", "G", "H", "J", "K", "L"]
    line3 = ["Z", "X", "C", "V", "B", "N", "M"]
    cv2.rectangle(frame,(x1,y1),(x2,y2),1,2)
    initial_x = copy.deepcopy(x1)
    initial_y = copy.deepcopy(y1)
    box_offset = (x2 - x1) / 10
    height_offset = (y2 - y1) / 3
    for i in range(10):  
        cv2.rectangle(frame,(initial_x, initial_y),(initial_x + box_offset, initial_y + height_offset),1,2)
        center_x = initial_x + (box_offset // 2)
        center_y = initial_y + (height_offset // 2)
        centroid_x.append(center_x)
        centroid_y.append(center_y)
        centroid_hash[line1[i]] = [center_x, center_y]
        find_centroid_for_each_letter(cv2, frame, center_x, center_y, line1[i])
        initial_x += box_offset
    initial_x = copy.deepcopy(x1) + box_offset // 2
    initial_y +=  height_offset
    for i in range(9):  
        cv2.rectangle(frame,(initial_x, initial_y),(initial_x + box_offset, initial_y + height_offset),1,2)
        center_x = initial_x + (box_offset // 2)
        center_y = initial_y + (height_offset // 2)
        centroid_x.append(center_x)
        centroid_y.append(center_y)
        centroid_hash[line2[i]] = [center_x, center_y]
        find_centroid_for_each_letter(cv2, frame, center_x, center_y, line2[i])
        initial_x += box_offset
    initial_x = copy.deepcopy(x1) + box_offset
    initial_y +=  height_offset
    for i in range(7):  
        cv2.rectangle(frame,(initial_x, initial_y),(initial_x + box_offset, initial_y + height_offset),1,2)
        center_x = initial_x + (box_offset // 2)
        center_y = initial_y + (height_offset // 2)
        centroid_x.append(center_x)
        centroid_y.append(center_y)
        centroid_hash[line3[i]] = [center_x, center_y]
        find_centroid_for_each_letter(cv2, frame, center_x, center_y, line3[i])
        initial_x += box_offset
    # print (centroid_x)
    # print (centroid_y)
    # print (len(centroid_x))
    # print (len(centroid_y))
    # print (centroid_hash)
    # exit()
    

def create_keyboard(cv2, frame, x1, y1, x2, y2):
    create_box_for_each_letter(cv2, frame, x1, y1, x2, y2)
    font_family = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    font_color = (255,0,0)
    font_thickness = 2
    letter_spacing = 5
    line_spacing = 50
    # cv2.putText(frame, "Q  W  E  R  T  Y  U  I  O  P", (300, 200), font_family, font_scale, font_color, font_thickness)
    # cv2.putText(frame, "  A  S  D  F  G  H  J  K  L  ", (300, 250), font_family, font_scale, font_color, font_thickness)
    # cv2.putText(frame, "    Z  X  C  V  B  N  M  ", (300, 300), font_family, font_scale, font_color, font_thickness)
    # find_centroid_for_each_letter(line1)
    # line_size = cv2.getTextSize("Q  W  E  R  T  Y  U  I  O  P  ", font_family, font_scale, font_thickness)
    # print (line_size)
    # line_size = cv2.getTextSize("I", font_family, font_scale, font_thickness)
    # print (line_size)
    # line_size = cv2.getTextSize("M", font_family, font_scale, font_thickness)
    # print (line_size)
    # line_size = cv2.getTextSize(" ", font_family, font_scale, font_thickness)
    # print (line_size)
    # line_size = cv2.getTextSize("  ", font_family, font_scale, font_thickness)
    # print (line_size)

def find_prediced_word(track_gesture_x, track_gesture_y):
    word = "Bannana"
    return word

def create_predicted_output(cv2, frame, word='Nil'):
    label = "Predicted : "
    font_family = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    font_color = (255,0,0)
    font_thickness = 2
    cv2.putText(frame, label + word, (100, 100), font_family, font_scale, font_color, font_thickness)

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video",
    help="path to the (optional) video file")
ap.add_argument("-b", "--buffer", type=int, default=64,
    help="max buffer size")
args = vars(ap.parse_args())

# define the lower and upper boundaries of the "green"
# ball in the HSV color space, then initialize the
# list of tracked points
greenLower = (29, 86, 6)
greenUpper = (64, 255, 255)
pts = deque(maxlen=args["buffer"])

# if a video path was not supplied, grab the reference
# to the webcam
if not args.get("video", False):
    vs = VideoStream(src=0).start()

# otherwise, grab a reference to the video file
else:
    vs = cv2.VideoCapture(args["video"])   

# allow the camera or video file to warm up
time.sleep(2.0)

# img_path = 'qwerty-min.png'
# logo = cv2.imread(img_path, -1)
# logo = cv2.flip(logo, 1)
# watermark = image_resize(logo, height=160)
# watermark = cv2.cvtColor(watermark, cv2.COLOR_BGR2BGRA)
predicted_word_flag = False
startTimer = None
endTimer = None
word = ''
# keep looping
while True:
    if startTimer is not None:
        endTimer =  time.time()
        elapsed_time = endTimer - startTimer
        # print (elapsed_time)
        # print (elapsed_time > 5)
        if elapsed_time > 5:
            startTimer = None
            endTimer = None
            predicted_word_flag = False
    # grab the current frame
    frame = vs.read()
    frame = cv2.flip(frame, 1)

    # resize the frame, blur it, and convert it to the HSV
    # color space
    frame = imutils.resize(frame, width=1300)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)
    # frame_h, frame_w, frame_c = frame.shape
    # # overlay with 4 channels BGR and Alpha
    # overlay = np.zeros((frame_h, frame_w, 4), dtype='uint8')
    # watermark_h, watermark_w, watermark_c = watermark.shape
    blurred = cv2.GaussianBlur(frame, (11, 11), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

    # get the ROI
    # roi = frame[top:bottom, right:left]

    # handle the frame from VideoCapture or VideoStream
    frame = frame[1] if args.get("video", False) else frame

    # if we are viewing a video and we did not grab a frame,
    # then we have reached the end of the video
    if frame is None:
        break

    # replace overlay pixels with watermark pixel values
    # for i in range(0, watermark_h):
    #     for j in range(0, watermark_w):
    #         if watermark[i,j][3] != 0:
    #             offset_h = 385
    #             offset_w = 285
    #             h_offset = frame_h - watermark_h - offset_h
    #             w_offset = frame_w - watermark_w - offset_w
    #             overlay[h_offset + i,w_offset+ j] = watermark[i,j]

    # construct a mask for the color "green", then perform
    # a series of dilations and erosions to remove any small
    # blobs left in the mask
    mask = cv2.inRange(hsv, greenLower, greenUpper)
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)

    # find contours in the mask and initialize the current
    # (x, y) center of the ball
    cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    center = None

    # only proceed if at least one contour was found
    if len(cnts) > 0:
        # find the largest contour in the mask, then use
        # it to compute the minimum enclosing circle and
        # centroid
        c = max(cnts, key=cv2.contourArea)
        ((x, y), radius) = cv2.minEnclosingCircle(c)
        M = cv2.moments(c)
        center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
        # only proceed if the radius meets a minimum size
        if (radius > 10) and (x1 < center[0] < x2) and (y1 < center[1] < y2):
            # print ("Ball center")
            # print (center)
            # print (int(x), int(y))
            predicted_word_flag = False
            track_gesture_x.append(center[0])
            track_gesture_y.append(center[1])

            # draw the circle and centroid on the frame,
            # then update the list of tracked points
            cv2.circle(frame, (int(x), int(y)), int(radius),
                (0, 255, 255), 2)
            cv2.circle(frame, center, 2, (0, 0, 255), -1)
        # else:
        #     if len(track_gesture_x) > 0:
        #         print ("The gesture is")
        #         print (track_gesture_x)
        #         print (track_gesture_y)
        #     track_gesture_x = []
        #     track_gesture_y = []
    else:
        if len(track_gesture_x) > 0:
            print ("The gesture is")
            print (track_gesture_x)
            print (track_gesture_y)
            word = shark2Algorithm.predict_word(track_gesture_x, track_gesture_y)
            predicted_word_flag = True
            startTimer =  time.time()
        track_gesture_x = []
        track_gesture_y = []

    # update the points queue
    pts.appendleft(center)

    # loop over the set of tracked points
    for i in range(1, len(pts)):
        # if either of the tracked points are None, ignore
        # them
        if pts[i - 1] is None or pts[i] is None:
            continue

        # otherwise, compute the thickness of the line and
        # draw the connecting lines
        thickness = int(np.sqrt(args["buffer"] / float(i + 1)) * 2.5)
        cv2.line(frame, pts[i - 1], pts[i], (0, 0, 255), thickness)

    # cv2.addWeighted(overlay, 0.25, frame, 1.0, 0, frame)
    # show the frame to our screen
    create_keyboard(cv2, frame, x1, y1, x2, y2)
    if predicted_word_flag:
        create_predicted_output(cv2, frame, word)
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    # if the 'q' key is pressed, stop the loop
    if key == ord("q"):
        break

# if we are not using a video file, stop the camera video stream
if not args.get("video", False):
    vs.stop()

# otherwise, release the camera
else:
    vs.release()

# close all windows
cv2.destroyAllWindows()