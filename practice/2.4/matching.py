import cv2
import numpy as np


def main():

    cap = cv2.VideoCapture(0)
    while True:
        ret, img = cap.read()
        ret2, img2 = cap.read()

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

        # Transform second image. ##########################
        rows, cols = gray2.shape
        M = cv2.getRotationMatrix2D((cols/2, rows/2), 20, 1)
        img2 = cv2.warpAffine(img2, M, (cols, rows))
        pts1 = np.float32([[50, 50], [200, 50], [50, 200]])
        pts2 = np.float32([[10, 100], [200, 50], [100, 250]])
        M = cv2.getAffineTransform(pts1, pts2)
        img2 = cv2.warpAffine(img2, M, (cols, rows))
        #####################################################

        # ORB
        orb = cv2.ORB_create(50)

        kp, des = orb.detectAndCompute(img, None)
        kp2, des2 = orb.detectAndCompute(img2, None)
        matcher = cv2.DescriptorMatcher_create(
            cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)

        matches = matcher.match(des, des2, None)

        matches = sorted(matches, key=lambda x: x.distance)

        img3 = cv2.drawMatches(img, kp, img2, kp2, matches[:50], None)

        img = cv2.drawKeypoints(
            img, kp, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        img2 = cv2.drawKeypoints(
            img2, kp2, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

        cv2.imshow('Original', img3)
        #cv2.imshow('Transformed', img2)

        if cv2.waitKey(1) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
