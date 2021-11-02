import cv2
import numpy as np


def rotate_box(bb, cx, cy, h, w, theta):
    new_bb = list(bb)
    for i, coord in enumerate(bb):
        # opencv calculates standard transformation matrix
        M = cv2.getRotationMatrix2D((cx, cy), theta, 1.0)
        # Grab  the rotation components of the matrix)
        cos = np.abs(M[0, 0])
        sin = np.abs(M[0, 1])
        # compute the new bounding dimensions of the image
        nW = int((h * sin) + (w * cos))
        nH = int((h * cos) + (w * sin))
        # adjust the rotation matrix to take into account translation
        M[0, 2] += (nW / 2) - cx
        M[1, 2] += (nH / 2) - cy
        # Prepare the vector to be transformed
        v = [coord[0], coord[1], 1]
        # Perform the actual rotation and return the image
        calculated = np.dot(M, v)
        new_bb[i] = (calculated[0], calculated[1])
    return new_bb


def rotate_bound(image, angle):
    # grab the dimensions of the image and then determine the
    # centre
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)

    # grab the rotation matrix (applying the negative of the
    # angle to rotate clockwise), then grab the sine and cosine
    # (i.e., the rotation components of the matrix)
    M = cv2.getRotationMatrix2D((cX, cY), angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    # compute the new bounding dimensions of the image
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))

    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY

    # perform the actual rotation and return the image
    return cv2.warpAffine(image, M, (nW, nH))


def topRow(bb):
    # sort list if tuples by y value
    bb.sort(key=lambda x: x[1])
    # select smallest 2 coordinates
    bb = bb[0:2]
    # sort top row by x value
    bb.sort(key=lambda x: x[0])

    return bb


def Corners(bb):
    xs = list(set([round(i[0]) for i in bb]))
    ys = list(set([round(i[1]) for i in bb]))
    return min(xs), max(xs), min(ys), max(ys)
