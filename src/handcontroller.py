import cv2
import numpy as np
import pyautogui
# import os
# import sys
from PIL import Image
import math
from keras.preprocessing.image import img_to_array
from skimage.transform import rotate
from skimage.measure import regionprops
from keras.models import load_model
import plaidml.keras
plaidml.keras.install_backend()


def importMovie():
    cap = cv2.VideoCapture('datasets/movies/output5.mov')
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
            x, y, w, h, size = contours[nlabel]
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
    h = dst.shape[0]
    w = dst.shape[1]
    center = (int(w/2), int(h/2))
    trans = cv2.getRotationMatrix2D(center, 90, 1.0)
    flipdst = cv2.warpAffine(dst, trans, (w, h))

    return dst, flipdst


def getPatternMatch(dst, img):
    points = cv2.matchTemplate(dst, img, cv2.TM_CCOEFF_NORMED)

    w = dst.shape[1]
    h = dst.shape[0]

    return points, w, h


def onChangeFinger(num):
    pyautogui.typewrite(num)


def decision(editedImage, model, frame):
    # dst = cv2.inRange(editedImage, 100, 175)
    # closing = cv2.morphologyEx(dst, cv2.MORPH_CLOSE, ckernel)
    # opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, okernel)

    _, maxBlob = getMaximumBlob(editedImage)
    contours, hierarchy = cv2.findContours(maxBlob, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    im = rotateImg(maxBlob)
    recrop = handcrop(im)

    image = Image.fromarray(np.uint8(recrop))
    image = image.convert("L")
    image = image.resize((36, 36))
    image = np.asarray(image)
    img_nad = img_to_array(image)/255
    img_nad = img_nad[None, ...]

    label = ['finger0', 'finger1', 'finger2', 'finger3', 'finger4', 'finger5']
    pred = model.predict(img_nad, batch_size=1, verbose=0)
    score = np.max(pred)
    pred_label = label[np.argmax(pred[0])]
    cv2.putText(frame, f'{pred_label}:{score}', (0, 50), font, 2, (255, 255, 255), 2, cv2.LINE_AA)
    return pred_label, score


def rotateImg(image):
    # label_img = label(image)
    # print(label_img.shape)
    regions = regionprops(image, coordinates='xy')

    for props in regions:
        y0, x0 = props.centroid
        orientation = props.orientation
        # print(orientation)
        # x1 = x0 + math.cos(orientation) * 0.5 * props.major_axis_length
        # y1 = y0 - math.sin(orientation) * 0.5 * props.major_axis_length
        # x2 = x0 - math.sin(orientation) * 0.5 * props.minor_axis_length
        # y2 = y0 - math.cos(orientation) * 0.5 * props.minor_axis_length

    if np.argmax(image) > 0:
        image = rotate(image, angle=-(math.pi/2+orientation)*(180/math.pi), order=0, resize=True)
    else:
        image = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)

    return image


def handcrop(image):
    image = np.uint8(image)
    distance = cv2.distanceTransform(image, cv2.DIST_L2, 5)
    maxv = np.unravel_index(np.argmax(distance), distance.shape)

    # underlim = distance[maxv]*0.99
    # minv = distance[maxv]
    # mindist = np.max(distance)
    recrop = image[0:maxv[0], 0:image.shape[1]]
    _, recrop = getMaximumBlob(recrop)
    contours, hierarchy = cv2.findContours(recrop, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) > 0:
        x, y, w, h = cv2.boundingRect(contours[0])

        # 画像を切り抜いて正方形に変形
        length = max(w, h)
        relocate = np.zeros((length, length), dtype=np.uint8)
        hand_region = recrop[y:y+h, x:x+w]
        offset_y = length - hand_region.shape[0]
        offset_x = int((length - hand_region.shape[1]) / 2)
        relocate[offset_y:, offset_x:offset_x+hand_region.shape[1]] = hand_region

        return relocate
    else:
        return np.zeros((36, 36), dtype=np.uint8)


if __name__ == '__main__':
    cap = importMovie()
    prevnumfinger = 0
    rects = []
    model = load_model('/Users/suzukiyuusatoru/Documents/notebook/handdetect.h5')
    ckernel = np.ones((3, 3), np.uint8)
    okernel = np.ones((5, 5), np.uint8)
    font = cv2.FONT_HERSHEY_SIMPLEX
    while(cap.isOpened):
        ret, frame = cap.read()

        thresh = imgPreprocessing(frame)
        maxSize, maxBlob = getMaximumBlob(thresh)
        # closing = denoising(maxBlob)
        closing = cv2.morphologyEx(maxBlob, cv2.MORPH_CLOSE, ckernel)
        opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, okernel)
        pred_label, score = decision(opening, model, frame)

        cv2.imshow('frame', frame)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
        if cv2.waitKey(10) & 0xFF == ord('s'):
            cv2.imwrite('finger1.png', frame)

    cap.release()
    cv2.destroyAllWindows()