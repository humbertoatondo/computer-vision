import cv2
import numpy as np
import argparse


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
        minNeighbors=5,     # Objects detected near the current object.
        minSize=(30, 30),   # Size for each window.
        flags=cv2.CASCADE_SCALE_IMAGE  # cv2.CASCADE_SCALE_IMAGE.CV_HAAR_SCALE_IMAGE
    )

    profileFaces = profileFaceCascade.detectMultiScale(
        grayImage,          # Gray scale image.
        scaleFactor=1.1,    # Help detect faces from different distances.
        minNeighbors=3,     # Objects detected near the current object.
        minSize=(30, 30),   # Size for each window.
        flags=cv2.CASCADE_SCALE_IMAGE  # cv2.CASCADE_SCALE_IMAGE.CV_HAAR_SCALE_IMAGE
    )

    # Concatenate results and give corresponding shape.
    faces = np.append(frontalFaces, profileFaces)

    return np.reshape(faces, [-1, 4])


def blurGaussian(faces, image, mask):
    """
    Apply gaussian blur to regions of interest.

    Parameters:
    faces (numpy.ndarray):
        A bidimensional array containing regions of interest (x, y, w, h).
    image (numpy.ndarray):
        A bidimensional array containing image's pixel data.
    mask (numpy.ndarray):
        A bidimensional array describing the image's mask.

    Returns:
        tuple with updated image and mask
    """

    for (x, y, w, h) in faces:
        y -= 25
        h += 50

        ROI = image[y:y+h, x:x+w]
        blur = cv2.GaussianBlur(ROI, (91, 91), 0)

        image[y:y+h, x:x+w] = blur

        cv2.ellipse(
            mask, ((int((x + x + w)/2), int((y + y + h)/2)), (w, h), 0), 255, -1)

    return (image, mask)


def blurPixelate(faces, image, mask):
    """
    Apply pixelate filter to regions of interest.

    Parameters:
    faces (numpy.ndarray):
        A bidimensional array containing regions of interest (x, y, w, h).
    image (numpy.ndarray):
        A bidimensional array containing image's pixel data.
    mask (numpy.ndarray):
        A bidimensional array describing the image's mask.

    Returns:
        tuple with updated image and mask
    """

    for (x, y, w, h) in faces:
        y -= 25
        h += 50

        ROI = image[y:y+h, x:x+w]
        temp = cv2.resize(ROI, (16, 16), interpolation=cv2.INTER_LINEAR)
        blur = cv2.resize(temp, (w, h), interpolation=cv2.INTER_NEAREST)

        image[y:y+h, x:x+w] = blur

        cv2.ellipse(
            mask, ((int((x + x + w)/2), int((y + y + h)/2)), (w, h), 0), 255, -1)

    return (image, mask)


def applyBorders(faces, image):
    """
    Apply borders to regions of interest.

    Parameters:
    faces (numpy.ndarray):
        A bidimensional array containing regions of interest (x, y, w, h).
    image (numpy.ndarray):
        A bidimensional array containing image's pixel data.
    """

    for (x, y, w, h) in faces:
        y -= 25
        h += 50
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)


def runProgram(image, frontalFaceCascade, profileFaceCascade, argv):
    """
    Function responsible of running the main process.

    Parameter:
    image (numpy.ndarray):
        A bidimensional array containing image's pixel data.
    frontalFaceCascade (cv2.CascadeClassifier):
        A classifier that represents frontal face data.
    profileFaceCascade (cv2.CascadeClassifier):
        A classifier that represents profile face data.
    argv (argparse):
        Command line arguments.
    """
    # Preprocess image by applying a Gaussian Blur.
    blurredImage = cv2.GaussianBlur(image, (7, 7), 0)

    # Convert image to gray scale.
    gray = cv2.cvtColor(blurredImage, cv2.COLOR_BGR2GRAY)

    # Detect faces using Haar algorithm.
    faces = detectFaces(gray, frontalFaceCascade, profileFaceCascade)

    # Create a clone of our current frame and get its mask.
    tempImg = image.copy()
    maskShape = (image.shape[0], image.shape[1], 1)
    mask = np.full(maskShape, 0, dtype=np.uint8)

    # Select filter for ROIs.
    try:
        if (argv.gaussian):
            tempImg, mask = blurGaussian(faces, tempImg, mask)
        elif (argv.border):
            applyBorders(faces, image)
        else:
            tempImg, mask = blurPixelate(faces, tempImg, mask)
    except:
        pass

    # Apply resulting mask to our image and get results.
    mask_inv = cv2.bitwise_not(mask)
    img1_bg = cv2.bitwise_and(image, image, mask=mask_inv)
    img2_fg = cv2.bitwise_and(tempImg, tempImg, mask=mask)
    dst = cv2.add(img1_bg, img2_fg)

    # Display results.
    cv2.imshow('Display', dst)


def main():
    # Initialize argparse and add arguments.
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--image", help="path to image file")
    parser.add_argument(
        "-v", "--video", help="run program on live video", action="store_true")
    parser.add_argument(
        "-g", "--gaussian", help="apply gaussian blur to detected faces", action="store_true")
    parser.add_argument(
        "-p", "--pixelate", help="apply pixelate filter to detected faces", action="store_true")
    parser.add_argument(
        "-b", "--border", help="apply a rectangular border over faces", action="store_true")

    argv = parser.parse_args()

    # Get Haar cascades.
    frontalFaceCascade = cv2.CascadeClassifier("haarcascade_frontalface.xml")
    profileFaceCascade = cv2.CascadeClassifier("haarcascade_profileface.xml")

    if (argv.video):
        cap = cv2.VideoCapture(0)
        while True:
            ret, image = cap.read()

            runProgram(image, frontalFaceCascade, profileFaceCascade, argv)

            if cv2.waitKey(1) == 27:
                break

        cap.release()
        cv2.destroyAllWindows()

    elif (argv.image):
        image = cv2.imread(argv.image)

        runProgram(image, frontalFaceCascade, profileFaceCascade, argv)

        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
