# Tracking motion from the video and generating
# the UV of the object related to the frame

import cv2
import numpy as np
import math
import os


SHAPE = [400, 600]


# Detect frame

def get_line(image, shape):
    """Get two horitontal or vertical lines
        (40, 1): horizontal
        (1, 40): vertical
        Reference: https://stackoverflow.com/questions/7227074/horizontal-line-detection-with-opencv
    """
    thresh = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 51, 9)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, shape)
    detect = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    cont = cv2.findContours(detect, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cont = cont[0] if len(cont) == 2 else cont[1]
    def center(ps):
        return np.mean(ps, axis=1)[0]
    def loss(ps):
        dx = np.ptp(ps[:,0,0])
        dy = np.ptp(ps[:,0,1])
        mean = center(ps)
        lmean = np.linalg.norm(mean-[0.5*SHAPE[0],0.5*SHAPE[1]])
        return 1./dy+lmean
    if shape[0] > shape[1]:  # horizontal
        x1 = [c for c in cont if center(c)[1] < 0.4*SHAPE[1]]
        x2 = [c for c in cont if center(c)[1] > 0.6*SHAPE[1]]
        return [
            min(x1, key=lambda x: loss(x)),
            min(x2, key=lambda x: loss(x))
        ]
    else:  # vertical
        x1 = [c for c in cont if center(c)[0] < 0.4*SHAPE[0]]
        x2 = [c for c in cont if center(c)[0] > 0.6*SHAPE[0]]
        return [
            min(x1, key=lambda x: loss(x)),
            min(x2, key=lambda x: loss(x))
        ]


def linreg_ortho(contour):
    """Fit contour to a straight line with equation ax+by=c
        Returns a, b, c
        Reference: https://www.desmos.com/calculator/cs4faizltl
    """
    n = len(contour)
    x = contour[:, 0, 0]
    y = contour[:, 0, 1]
    m = n * sum(x*y) - sum(x)*sum(y)
    k = (n*sum(x**2) - n*sum(y**2) + sum(y)**2 - sum(x)**2) / m
    a = k - np.sign(m) * math.hypot(k, 2)
    b = 2
    c = (a*sum(x)+b*sum(y))/n
    m = 1.0 / math.hypot(a, b)
    return m*a, m*b, m*c


def line_intersect(a1, b1, c1, a2, b2, c2):
    """solve the linear system
        a1 x + b1 y = c1
        a2 x + b2 y = c2
    """
    m = np.array([[a1, b1], [a2, b2]], dtype=np.float64)
    b = np.array([c1, c2], dtype=np.float64)
    return np.linalg.inv(m).dot(b)


def get_frame(image):
    """Get the coordinates of vertices of the quadrilateral frame
        Returns a list of 4 points clockwise starting from the top left
    """
    hor = get_line(image, (40, 1))
    ver = get_line(image, (1, 40))
    #cv2.polylines(image, hor+ver, True, (0, 0, 0), 2)
    #cv2.imshow('Frame',image);cv2.waitKey(0)
    pnts = []
    for ch in hor:
        for cv in ver:
            p = line_intersect(*linreg_ortho(ch), *linreg_ortho(cv))
            pnts.append(p)
    pnts.sort(key=lambda p: math.atan2(p[1]-0.5*SHAPE[1], p[0]-0.5*SHAPE[0]))
    return np.array(pnts, dtype=np.float64)


def clip_frame(image, pnts_frame):
    """Clip the image based on the frame
        Reference: https://answers.opencv.org/question/231798/how-to-do-a-perspective-transformation-of-an-image-which-is-missing-corners-using-opencv-java/
    """
    pnts_dist = np.array([
        [0, 0],
        [SHAPE[0], 0],
        [SHAPE[0], SHAPE[1]],
        [0, SHAPE[1]]
    ])
    h, status = cv2.findHomography(pnts_frame, pnts_dist)
    image = cv2.warpPerspective(image, h, SHAPE)
    return image


# Detect object

def get_object_pos(image):
    thresh = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 51, 9)
    cont = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cont = cont[0] if len(cont) == 2 else cont[1]
    def loss(ps):
        dx = np.ptp(ps[:,0,0])
        dy = np.ptp(ps[:,0,1])
        area = cv2.contourArea(ps)
        square = abs(dx-dy) / (math.sqrt(dx*dy)+1)
        size = abs(math.sqrt(area)-SHAPE[0]*0.04)
        return square + size
    cont = sorted(list(cont), key=lambda x: loss(x))[0]
    return np.mean(cont, axis=0)[0]


def get_obj_pos_main(frame):
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    pnts = get_frame(image)
    clipped = clip_frame(image, pnts)
    objpos = get_object_pos(clipped)
    return objpos/SHAPE


# Video

def track_video(filename):
    cap = cv2.VideoCapture(filename)

    points = []

    fps = cap.get(cv2.CAP_PROP_FPS)
    dt = 1.0 / fps  # 1/120
    skip = max(fps//30, 1)

    framei = 0
    while cap.isOpened():
        flag, frame = cap.read()
        if not flag:
            break
        framei += 1
        if (framei-1) % skip != 0:
            continue
        if (framei//skip) % 100 == 0:
            print(framei//skip)
        t = framei * dt

        try:
            frame = cv2.resize(frame, SHAPE)
            pos = get_obj_pos_main(frame)
            points.append((t, pos[0], pos[1]))
        except:
            pass
        #print('('+','.join(["{:.3f}".format(x) for x in pos]),end='),')

        if False:  # debug
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            pnts = get_frame(image)
            clipped = clip_frame(image, pnts)
            objpos = get_object_pos(clipped)
            cv2.circle(clipped, objpos.astype(np.int32), SHAPE[0]//25, (0, 0, 0), 2)
            cv2.imshow('Frame',clipped)
            cv2.waitKey(0)
            #break

        #if framei//skip > 100: break

    cap.release()
    return points


lengths = [80, 70, 60, 50, 40, 30, 20]
for l in lengths:
    print(l)
    data = track_video(f'videos/{l}.mp4')
    with open(f'csv/uvt-{l}.csv', 'w') as fp:
        for p in data:
            print(','.join(map(str, p)), file=fp)
