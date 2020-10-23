import cv2
import numpy as np


def main():

    cap = cv2.VideoCapture(0)
    while True:
        ret, img = cap.read()

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# ----------------- Key Point Detectors ------------------

        # Harris --------------------------------------------
        # gray = np.float32(gray)
        # harris = cv2.cornerHarris(gray, 2, 3, 0.04)
        # img[harris > 0.01 * harris.max()] = [255, 0, 0]

        # Good Features to Track ----------------------------
        # corners = cv2.goodFeaturesToTrack(gray, 50, 0.01, 10)
        # corners = np.int0(corners)
        # for i in corners:
        #     x, y = i.ravel()
        #     cv2.circle(img, (x, y), 3, 255, -1)

        # FAST ----------------------------------------------
        # detector = cv2.FastFeatureDetector_create(50)
        # kp = detector.detect(img, None)
        # img2 = cv2.drawKeypoints(img, kp, None, flags=0)

        # cv2.imshow("FAST", img2)
        # if cv2.waitKey(1) == 27:
        #     break

# ----------------- Descriptors ------------------
        # BRIEF

        # ORB
        orb = cv2.ORB_create(50)
        kp, des = orb.detectAndCompute(img, None)
        img2 = cv2.drawKeypoints(
            img, kp, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

        cv2.imshow('ORB', img2)

        if cv2.waitKey(1) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
