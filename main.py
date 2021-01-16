import cv2
import numpy as np
from model.handtrack_tflite import HandTracker
from draw import DrawRec, DrawHand

# 旧的palm检测器更加稳，但是速度也会慢一些
old_palm=False
palm_path = "model/palm_detection" +("_old.tflite" if old_palm else ".tflite")
anchor_csv = "model/anchors"+ ("_old.csv" if old_palm else ".csv")
hand_path = "model/hand_landmark.tflite"
handDet = HandTracker(palm_path, hand_path, anchor_csv, 256 if old_palm else 128, 224)

capture = cv2.VideoCapture("test/testvideo2.mp4")
cv2.namedWindow("test", cv2.WINDOW_NORMAL)

while 1:
    ret, frame = capture.read()
    if not ret:
        break
    # frame=cv2.imread("test/testim.jpg")
    # 镜像图片
    # frame = cv2.flip(frame, 1)
    # 检测手势
    keypoints3d, bbox, o_1, o_2 = handDet(frame)
    if bbox is not None:
        print("hand flag:{0:.6f} hand side:{1:.6f}".format(o_1, o_2))
        frame = DrawRec(frame, bbox)
        frame = DrawHand(frame, keypoints3d)
    cv2.imshow("test", frame)
    cv2.waitKey(10)