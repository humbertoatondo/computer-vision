import cv2
import numpy as np


def main():

    cap = cv2.VideoCapture(0)
    while True:
        ret, img = cap.read()

        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        blur = cv2.GaussianBlur(gray, (7, 7), 0)

        # originalImage = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        originalImage = cv2.cvtColor(blur, cv2.COLOR_GRAY2RGB)
        reshapedImage = np.float32(originalImage.reshape(-1, 3))

        numberOfClusters = 6

        stopCriteria = (cv2.TERM_CRITERIA_MAX_ITER, 5, 0.1)

        ret, labels, clusters = cv2.kmeans(
            reshapedImage, numberOfClusters, None, stopCriteria, 2, cv2.KMEANS_RANDOM_CENTERS)

        clusters = np.uint8(clusters)

        intermediateImage = clusters[labels.flatten()]
        clusteredImage = intermediateImage.reshape((originalImage.shape))

        cv2.imshow("Video", originalImage)
        cv2.imshow("K-Means", clusteredImage)

        if cv2.waitKey(1) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
