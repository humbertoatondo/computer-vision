import cv2
import numpy as np


def detectFaces(grayImage, frontalFaceCascade, profileFaceCascade):
    """
    Detect frontal and profile faces using Haar algorithm.

    Parameters:
    grayImage (numpy.ndarray):
        A bidimensional array that represents pixel values
        in a gray scale image.
    frontalFaceCascade (cv2.CascadeClassifier):
        A classifier that represents frontal face data.
    profileFaceCascade (cv2.CascadeClassifier):
        A classifier that represents profile face data.

    Returns:
        numpy.ndarray: position (x, y) and dimensions (w, h) of detected images.
    """

    frontalFaces = frontalFaceCascade.detectMultiScale(
        grayImage,          # Gray scale image.
        scaleFactor=1.1,    # Help detect faces from different distances.
        # Objects detected near the current one before it declares a that a face was found.
        minNeighbors=5,
        minSize=(30, 30),   # Size for each window.
        flags=cv2.CASCADE_SCALE_IMAGE  # cv2.CASCADE_SCALE_IMAGE.CV_HAAR_SCALE_IMAGE
    )

    profileFaces = profileFaceCascade.detectMultiScale(
        grayImage,          # Gray scale image.
        scaleFactor=1.1,    # Help detect faces from different distances.
        # Objects detected near the current one before it declares a that a face was found.
        minNeighbors=3,
        minSize=(30, 30),   # Size for each window.
        flags=cv2.CASCADE_SCALE_IMAGE  # cv2.CASCADE_SCALE_IMAGE.CV_HAAR_SCALE_IMAGE
    )

    # Concatenate results and give corresponding shape.
    faces = np.append(frontalFaces, profileFaces)

    return np.reshape(faces, [-1, 4])


def main():
    # Create Haar cascade.
    frontalFaceCascade = cv2.CascadeClassifier("haarcascade_frontalface.xml")
    profileFaceCascade = cv2.CascadeClassifier("haarcascade_profileface.xml")

    cap = cv2.VideoCapture(0)
    while True:
        ret, image = cap.read()
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Detect faces using Haar algorithm.
        faces = detectFaces(gray, frontalFaceCascade, profileFaceCascade)

        tempImg = image.copy()
        maskShape = (image.shape[0], image.shape[1], 1)
        mask = np.full(maskShape, 0, dtype=np.uint8)
        # Draw a rectangle around detected faces.
        for (x, y, w, h) in faces:
            x = int(x * 1)
            y = int(y * 0.75)
            w = int(w * 1)
            h = int(h * 1.35)

            # tempImg[y:y+h, x:x+w] = cv2.blur(tempImg[y:y+h, x:x+w], (60, 60))
            ROI = tempImg[y:y+h, x:x+w]
            blur = cv2.GaussianBlur(ROI, (61, 61), 0)
            # temp = cv2.resize(ROI, (24, 24), interpolation=cv2.INTER_LINEAR)
            # blur = cv2.resize(temp, (w, h), interpolation=cv2.INTER_NEAREST)

            tempImg[y:y+h, x:x+w] = blur
            # cv2.ellipse(
            #     tempImg, ((int((x + x + w)/2), int((y + y + h)/2)), (w, h), 0), 255, 5)
            cv2.ellipse(
                mask, ((int((x + x + w)/2), int((y + y + h)/2)), (w, h), 0), 255, -1)

        # oustide of the loop, apply the mask and save
        mask_inv = cv2.bitwise_not(mask)
        img1_bg = cv2.bitwise_and(image, image, mask=mask_inv)
        img2_fg = cv2.bitwise_and(tempImg, tempImg, mask=mask)
        dst = cv2.add(img1_bg, img2_fg)

        cv2.imshow('Faced found', dst)

        if cv2.waitKey(1) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
