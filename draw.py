import cv2
import numpy as np
#        8   12  16  20
#        |   |   |   |
#        7   11  15  19
#    4   |   |   |   |
#    |   6   10  14  18
#    3   |   |   |   |
#    |   5---9---13--17
#    2    \         /
#     \    \       /
#      1    \     /
#       \    \   /
#        ------0-
connections = [(0, 1), (1, 2), (2, 3), (3, 4), (5, 6), (6, 7), (7, 8), (9, 10),
               (10, 11), (11, 12), (13, 14), (14, 15), (15, 16), (17, 18),
               (18, 19), (19, 20), (0, 5), (5, 9), (9, 13), (13, 17), (0, 17)]


def DrawHand(frame, keypoints,pcolor=(0,0,225),lcolor=(0,0,225)):
    r"""
    绘制手部21个点
    """
    keypoints=keypoints.astype(np.int)
    for c1, c2 in connections:
        cv2.line(frame, tuple(keypoints[c1]), tuple(keypoints[c2]),
                 lcolor, 2, 8)
    for point in keypoints:
        cv2.circle(frame, tuple(map(int, point)), 3, pcolor, -1)
    return frame


def DrawRec(frame, bbox):
    r"""
    绘制手部包围框
    """
    cv2.polylines(frame, [bbox.astype(np.int32).reshape((-1, 1, 2))],
                  isClosed=True,
                  color=(0, 0, 255),
                  thickness=3,
                  lineType=8)
    return frame
