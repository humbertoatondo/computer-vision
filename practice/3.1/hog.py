import numpy as np
import cv2
import matplotlib.pyplot as plt


def main():
    # Load pre-trained classifiers.
    # face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    # eye_cascade = cv2.CascadeClassifier("haarcascade_eye.xml")

    # Load and transform image.
    img = cv2.imread("face.jpg")
    img = np.float32(img) / 255.0

    # Calculate gradient
    gx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=1)
    gy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=1)

    # Python Calculate gradient magnitude and direction ( in degrees )
    mag, angle = cv2.cartToPolar(gx, gy, angleInDegrees=True)

    arr = [0] * 9
    for i in range(0, img.shape[0]):
        for j in range(0, img.shape[1]):
            x = np.int((angle[i][j][0]) % 180 // 20)
            y = np.int((angle[i][j][1]) % 180 // 20)
            z = np.int((angle[i][j][2]) % 180 // 20)
            arr[x] += mag[i][j][0]
            arr[y] += mag[i][j][1]
            arr[z] += mag[i][j][2]

    for i in range(0, 9):
        print(arr[i])

    plt.bar([0, 1, 2, 3, 4, 5, 6, 7, 8], arr)
    plt.show()

    cv2.imshow("Original Image", img)
    cv2.imshow("Result", mag)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
