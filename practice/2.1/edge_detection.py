import cv2
import numpy as np
import sys


def main():
    method = "canny"
    if (len(sys.argv) > 1):
        if (sys.argv[1] == "prewitt" or sys.argv[1] == "sobel"):
            method = sys.argv[1]

    print(method)

    cap = cv2.VideoCapture(0)
    while True:
        ret, img = cap.read()

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (3, 3), 0)

        res = ""
        if (method == "canny"):
            res = cv2.Canny(blur, 10, 70)
        elif (method == "sobel"):
            img_sobelx = cv2.Sobel(blur, cv2.CV_8U, 1, 0, ksize=5)
            img_sobely = cv2.Sobel(blur, cv2.CV_8U, 0, 1, ksize=5)
            res = img_sobelx + img_sobely
        else:
            kernelx = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])
            kernely = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
            img_prewittx = cv2.filter2D(blur, -1, kernelx)
            img_prewitty = cv2.filter2D(blur, -1, kernely)
            res = img_prewittx + img_prewitty

        ret, mask = cv2.threshold(res, 70, 255, cv2.THRESH_BINARY)

        res = cv2.bitwise_and(img, img, mask=mask)

        cv2.imshow('Video feed', res)

        if cv2.waitKey(1) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
