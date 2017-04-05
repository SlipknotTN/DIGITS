import cv2
import numpy as np

def get5XCrops(image, debug = False):
    """
    Simple augmentation. Angles and center. No checks on form factor
    :param image:
    :param squareSize:
    :param debug:
    :return:
    """
    width = image.shape[1]
    height = image.shape[0]

    crops = []

    imageDebug = np.copy(image)

    # Upper Left
    crops.append(image[0:height/2, 0:width/2])

    # Upper right
    crops.append(image[0:height / 2, width / 2 + 1 : width - 1])

    # Bottom left
    crops.append(image[height / 2 + 1: height - 1, 0:width / 2])

    # Bottom right
    crops.append(image[height / 2 + 1: height - 1, width / 2 + 1 : width - 1])

    # Center
    crops.append(image[height / 4 * 1 + 1: height / 4 * 3, width / 4 * 1 + 1: width / 4 * 3])

    if debug:
        cv2.imshow("default", imageDebug)
        cv2.waitKey(0)
        for finalCrop in crops:
            cv2.imshow("crops", finalCrop)
            cv2.waitKey(0)

    return crops



def getMultipleCrops(image, squareSize, debug = False):
    """
    Sliding window crops every half square size. Crops are squares of squareSize
    """

    width = image.shape[1]
    height = image.shape[0]

    crops = []

    imageDebug = np.copy(image)

    # Iterate over vertical axis
    yCropIndex = 0
    while (True):

        startY = int(yCropIndex * float(squareSize)/2.0)
        finishY = startY + squareSize

        # Column terminates
        if finishY >= height:
            break;

        # Crops along the row (horizontal axis)
        xCropIndex = 0
        while (True):

            startX = int(xCropIndex * float(squareSize)/2.0)
            finishX = startX + squareSize

            # Row terminates
            if finishX >= width:
                break;

            cv2.rectangle(imageDebug, pt1=(startX, startY), pt2=(finishX, finishY),
                          color=(0, 0, 255), thickness=5)

            crops.append(image[startY:finishY, startX:finishX])

            xCropIndex += 1

        yCropIndex += 1

    if debug:
        cv2.imshow("default", imageDebug)
        cv2.waitKey(0)
        for finalCrop in crops:
            cv2.imshow("crops", finalCrop)
            cv2.waitKey(0)

    return crops