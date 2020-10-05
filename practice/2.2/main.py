import cv2
import numpy as np
import sys


def main():

    cap = cv2.VideoCapture(0)
    while True:
        ret, img = cap.read()

        hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        kernel = np.ones((5, 5), np.uint8)

        # Red color
        low_red = np.array([161, 155, 84])
        high_red = np.array([179, 255, 255])
        red_mask = cv2.inRange(hsv_img, low_red, high_red)
        red_mask = cv2.erode(red_mask, kernel, iterations=1)
        red_mask = cv2.dilate(red_mask, kernel, iterations=2)
        red = cv2.bitwise_and(img, img, mask=red_mask)

        # Blue color
        low_blue = np.array([94, 80, 2])
        high_blue = np.array([126, 255, 255])
        blue_mask = cv2.inRange(hsv_img, low_blue, high_blue)
        blue_mask = cv2.erode(blue_mask, kernel, iterations=3)
        blue_mask = cv2.dilate(blue_mask, kernel, iterations=3)
        blue = cv2.bitwise_and(img, img, mask=blue_mask)

        # Green color
        low_green = np.array([25, 52, 72])
        high_green = np.array([102, 255, 255])
        green_mask = cv2.inRange(hsv_img, low_green, high_green)
        green_mask = cv2.erode(green_mask, kernel, iterations=2)
        green_mask = cv2.dilate(green_mask, kernel, iterations=1)
        green = cv2.bitwise_and(img, img, mask=green_mask)

        cv2.imshow('Video feed', img)
        cv2.imshow("Red mask", red)
        cv2.imshow("Blue mask", blue)
        cv2.imshow("Green mask", green)

        if cv2.waitKey(1) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
