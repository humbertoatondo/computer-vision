from PIL import Image
import colorsys
import numpy as np
import argparse
import cv2
import matplotlib.pyplot as plt

def threshold_pixel(r, g, b):
    h, l, s = colorsys.rgb_to_hls(r / 255., g / 255., b / 255.)
    return 1 if l > .36 else 0

def hlsify(img):
    pixels = img.load()
    width, height = img.size

    # Create a new blank monochrome image.
    output_img = Image.new('1', (width, height), 0)
    output_pixels = output_img.load()

    for i in range(width):
        for j in range(height):
            output_pixels[i, j] = threshold_pixel(*pixels[i, j])

    return output_img

def pixel_frequencies(image):
    nums = []
    for i in range(len(image) - 1):
        for j in range(len(image[0] - 1)):
            nums.append(image[i][j])
            # if image[i][j] in dict:
            #     dict[image[i][j]] += 1
            # else:
            #     dict[image[i][j]] = 0
    return nums


def main():
    # construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", required=True,
        help="path to the input image")
    args = vars(ap.parse_args())
    # construct the Gaussian blur
    gauss = np.array((
        [1/9, 1/9, 1/9],
        [1/9, 1/9, 1/9],
        [1/9, 1/9, 1/9]
    ))

    # load the input image and convert it to grayscale
    image = cv2.imread(args["image"])
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    nums = pixel_frequencies(gray)
    plt.hist(nums, bins = 25)
    plt.show()


    print("[INFO] applying {} kernel".format("gauss"))
    opencvOutput = cv2.filter2D(gray, -1, gauss)
    # show the output images
    cv2.imshow("original", gray)
    #cv2.imshow("{} - convole".format("gauss"), convoleOutput)
    cv2.imshow("{} - opencv".format("gauss"), opencvOutput)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


    binarified_img = hlsify(Image.open(args["image"]))
    binarified_img.show()

if __name__ == "__main__":
    main()