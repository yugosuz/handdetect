import cv2
import numpy as np
import pyautogui
# import os
# import sys
# import math


def loadVideo(path):
    """
    動画を読み込む
    Args:
        path(str) : 動画へのパス
    Returns:
        cap : VideoCaptureクラス
    """
    cap = cv2.VideoCapture(path)
    return cap


def imgPreprocessing(frame):
    """
    画像を読み込み、二値化する
    Args:
        frame(numpy.ndarray) : 二値化する画像
    Returns:
        thresh(numpy.ndarray) : 二値化した画像
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    thresh = cv2.inRange(gray, 65, 170)
    return thresh


def getMaximumBlob(mask):
    """
    画像から最大のブロブを抜き出し、それ以外の領域を黒塗りする
    Args:
        mask(numpy.ndarray) : 抽出したい画像
    Returns:
        maxSize(float) : 取得したブロブのサイズ
        mask(numpy.ndarray) : 最大ブロブ以外を塗りつぶした画像
    """
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
    """
    ノイズ除去
    Args:
        blob(numpy.ndarray) : ノイズを消したい画像
    Returns:
        closing(numpy.ndarray) : ノイズ除去（クロージング）を施した画像
    """
    kernel = np.ones((5, 5), np.uint8)
    closing = cv2.morphologyEx(blob, cv2.MORPH_CLOSE, kernel)
    return closing


def getContours(img):
    """
    画像から凸包を計算する
    Args:
        img(numpy.ndarray) : 凸包を計算したい画像
    Returns:
        contours(numpy.ndarray) : 凸包の輪郭の座標
    """
    contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    return contours


def getPatternMatch(dst, img):
    """
    パターンマッチを行う
    Args:
        dst(numpy.ndarray) : 探索する画像
        img(numpy.ndarray) : テンプレート
    Returns:
        points(numpy.ndarray) : 類似度マップ
        w(int) : 画像のwidth
        h(int) : 画像のheight
    """
    points = cv2.matchTemplate(dst, img, cv2.TM_CCOEFF_NORMED)

    w = dst.shape[1]
    h = dst.shape[0]

    return points, w, h


def onChangeFinger(num):
    """
    認識した指の本数に対してその番号に対応するキーを押す
    Args:
        num(int) : 指の本数
    """
    pyautogui.typewrite(num)


if __name__ == '__main__':
    cap = loadVideo('../movies/output5.mov')
    rects = []
    ckernel = np.ones((3, 3), np.uint8)
    okernel = np.ones((5, 5), np.uint8)
    font = cv2.FONT_HERSHEY_SIMPLEX
    finger = cv2.imread('', cv2.IMREAD_GRAYSCALE)
    while(cap.isOpened):
        ret, frame = cap.read()

        thresh = imgPreprocessing(frame)
        maxSize, maxBlob = getMaximumBlob(thresh)
        closing = denoising(maxBlob)

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