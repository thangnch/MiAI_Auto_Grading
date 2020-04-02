import cv2
import math
import numpy as np
import random
from utils import *


# 1. Doc anh, chuyen thanh anh xam
image = cv2.imread("omr_test_02.png")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

# 2. Threshold de
thresh = cv2.adaptiveThreshold(blurred,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,            cv2.THRESH_BINARY_INV,31,3)
cv2.imshow("Anh tai buoc 2",thresh)
cv2.waitKey()


# 3. Tim khung ben ngoai de tach van ban khoi nen
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
contours = sorted(contours, key=lambda x: cv2.contourArea(x),reverse=True)
approx = cv2.approxPolyDP(contours[1], 0.01 * cv2.arcLength(contours[1], True), True)
rect = cv2.minAreaRect(contours[1])
box = cv2.boxPoints(rect);

# 4. Thuc hien transform de xoay van ban
corner = find_corner_by_rotated_rect(box,approx)
image = four_point_transform(image,corner)
wrap = four_point_transform(thresh,corner)
cv2.imshow("Anh sau buoc 4",wrap)
cv2.waitKey()

# 5. Tim cac o tick trong hinh
contours, _ = cv2.findContours(wrap, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
tickcontours = []
# loop over the contours
for c in contours:
    (x, y, w, h) = cv2.boundingRect(c)
    ar = w / float(h)
    if w >= 30 and h >= 30 and 0.8 <= ar <= 1.2:
        tickcontours.append(c)

# 6. So sanh cac o tick voi dap an

# Dinh nghia dap an
right_anser = {0: 1, 1: 4, 2: 0, 3: 3, 4: 1,5: 1,6: 0,7: 4}

# Sap xep cac contour theo hang
tickcontours = sort_contours(tickcontours, method="top-to-bottom")[0]

correct = 0

for (q, i) in enumerate(np.arange(0, len(tickcontours), 5)):

    # Dinh nghia mau rieng cho tung cau hoi
    color = (random.randint(0, 255),random.randint(0, 255),random.randint(0, 255))
    # Sap xep cac contour theo cot
    cnts = sort_contours(tickcontours[i:i + 5])[0]
    #cv2.drawContours(image, cnts, -1, color, 3)

    choice = (0,0)
    total = 0
    # Duyet qua cac contour trong hang
    for (j, c) in enumerate(cnts):

        # Tao mask de xem muc do to mau cua contour
        mask = np.zeros(wrap.shape, dtype="uint8")
        cv2.drawContours(mask, [c], -1, 255, -1)
        mask = cv2.bitwise_and(wrap, wrap, mask=mask)
        total = cv2.countNonZero(mask)

        # Lap de chon contour to mau dam nhat
        if total > choice[0]:
            choice = (total, j)

    # Lay dap an cua cau hien tai
    current_right = right_anser[q]
    # Kiem tra voi lua chon cua nguoi dung
    if current_right == choice[1]:
        # Neu dung Thi to mau xanh
        color = (0, 255, 0)
        correct += 1
    else:
        # Neu sai Thi to mau do
        color = (0, 0, 255)
    # Ve ket qua len anh
    cv2.drawContours(image, [cnts[current_right]], -1, color, 3)

drawText(image,'So cau tra loi dung:' + str(correct))

cv2.imshow("A",image)
cv2.waitKey()