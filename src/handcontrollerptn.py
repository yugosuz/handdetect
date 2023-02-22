import cv2
import numpy as np
import pyautogui
# import os
# import sys
# import math


def loadVideo(path):
    cap = cv2.VideoCapture(path)
    return cap


def imgPreprocessing(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    thresh = cv2.inRange(gray, 65, 170)
    return thresh


def getMaximumBlob(mask):
    nlabels, labelimg, contours, CoGs = cv2.connectedComponentsWithStats(mask)

    if nlabels > 0:
        maxLabel = 0
        maxSize = 0
        for nlabel in range(1, nlabels):
            x, y, w, h,size = contours[nlabel]
            xg, yg = CoGs[nlabel]
            if maxSize < size:
                maxSize = size
                maxLabel = nlabel

        # 最大領域を表すラベルを持つ画素値を255にする
        mask[labelimg == maxLabel] = 255
        # それ以外の領域はすべて0にする
        mask[labelimg != maxLabel] = 0

        return maxSize, mask
    else:
        return maxSize, mask


def denoising(blob):
    kernel = np.ones((5, 5), np.uint8)
    closing = cv2.morphologyEx(blob, cv2.MORPH_CLOSE, kernel)
    return closing


def getContours(img):
    contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    return contours


def preparePatternMatch():
    img = cv2.imread("/Users/suzukiyuusatoru/Documents/GitHub/HandTrack/Research/datasets/binalizeddatasets/test/finger1/picture533.png", cv2.IMREAD_GRAYSCALE)
    dst = img[180:225, 237:282]
    # h = dst.shape[0]
    # w = dst.shape[1]
    # center = (int(w/2), int(h/2))
    # trans = cv2.getRotationMatrix2D(center, 90 , 1.0)
    # flipdst = cv2.warpAffine(dst, trans, (w, h))
    # return dst, flipdst

    return dst


def getPatternMatch(dst, img):
    points = cv2.matchTemplate(dst, img, cv2.TM_CCOEFF_NORMED)

    w = dst.shape[1]
    h = dst.shape[0]

    return points, w, h


def onChangeFinger(num):
    pyautogui.typewrite(num)


if __name__ == '__main__':
    cap = loadVideo('datasets/movies/output5.mov')
    rects = []
    ckernel = np.ones((3, 3), np.uint8)
    okernel = np.ones((5, 5), np.uint8)
    font = cv2.FONT_HERSHEY_SIMPLEX
    while(cap.isOpened):
        ret, frame = cap.read()

        thresh = imgPreprocessing(frame)
        maxSize, maxBlob = getMaximumBlob(thresh)
        closing = denoising(maxBlob)

        finger = preparePatternMatch()
        dst, w, h = getPatternMatch(finger, closing)
        area = cv2.inRange(dst, 0.73, 1.0)

        contours, hierarchy = cv2.findContours(area, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            x, y, rw, rh = cv2.boundingRect(cnt)
            bottom_right = (x + rw + w, y + rh + h)
            cv2.rectangle(frame, (x + rw, y + rh), bottom_right, (0, 255, 0), 2)
            cv2.putText(frame, str(len(contours)), (10, 50), cv2.FONT_HERSHEY_PLAIN, 5, (0, 0, 255))

        cv2.imshow('frame', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        elif cv2.waitKey(1) & 0xFF == ord('s'):
            cv2.imwrite('finger0.png', frame)

    cap.release()
    cv2.destroyAllWindows()