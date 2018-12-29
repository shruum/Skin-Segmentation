import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
import glob
import os, os.path
from matplotlib import pyplot as plt
import argparse

class SkinSegmentor:

    def __init__(self):
        pass

    # Plot the image on MatPlotLib plot
    def plt_display(self, pltId, pltTitle, img, mode=0):
        # Choose subplot
        plt.subplot(pltId)
        plt.axis('off')

        # Choose Display mode
        if mode == 1:
            # Display image as grayscale
            img_to_display = img
            plt.imshow(img_to_display, 'gray')
        else:
            # Convert the image from BGR TO RGB
            # MatPlotLib requires the color images to be in RGB
            img_to_display = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            plt.imshow(img_to_display)
        plt.title(pltTitle)
        # to hide tick values on X and Y axis
        plt.xticks([])
        plt.yticks([])

    def display(self):
        #plt.tight_layout(h_pad=0.5)
        plt.draw()
        plt.waitforbuttonpress(0)
        plt.cla()

    # Creating a mask by thresholding for pixels in HSV color space
    def hsv_threshold(self, img):
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        lower = np.array([0,65,0])
        upper = np.array([120,150,255])
        mask = cv2.inRange(hsv, lower, upper);
        return mask

    # Creating a mask by thresholding for pixels in YCbCr color space
    def ycbcr_threshold(self, img):
        ycbcr = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
        lower = np.array([0,77,133])
        upper = np.array([255,127,173])
        mask = cv2.inRange(ycbcr, lower, upper);
        return mask

    # Applying Open and Close operations to fill in some holes and have a smoother mask
    def morphology(self, img):
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
        mask = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        return mask

    # Creating markers and doing watershed segmentation
    def watershed(self, src, mask):

        front = cv2.erode(mask, None, iterations = 5)
        back = cv2.dilate(mask, None, iterations = 5)
        unknown = back - front

        marker = np.zeros(mask.shape, dtype='int32')
        marker[front == 255] = 255
        marker[back == 0] = 128
        marker[unknown == 255] = 0
        #self.plt_display(223, 'Watershed Marker', marker, 1)

        ws_marker = cv2.watershed(src, marker)
        ws_marker = cv2.convertScaleAbs(ws_marker)
        ws_marker[ ws_marker == 128 ] = 0
        return ws_marker

    # The main functiion
    def segment(self, src):

        # Approach1 : Thresholding in color spaces
        hsv_mask = self.hsv_threshold(src)
        ycbcr_mask = self.ycbcr_threshold(src)
        mask = cv2.bitwise_or(hsv_mask, ycbcr_mask)
        #mask = self.morphology(mask)
        self.plt_display(221, 'Original', src)

        # Approach2 : Applying watershed algorithm
        mask = self.watershed(src, mask)

        self.plt_display(223, 'Watershed Mask', mask, 1)
        self.plt_display(224, 'Masked Original', cv2.bitwise_and(src, src, mask=np.uint8(mask)))

        return mask

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("imageFolder")
    args = parser.parse_args()
    imageFolder = args.imageFolder

    # Creating the Skin Segmentor object
    seg = SkinSegmentor()

    # Traversing through all the files in the images folder
    for filename in glob.glob(imageFolder + '/*'):
        print("Segmenting:", filename)
        src = cv2.imread(filename)

        # The main functin, returns the output "mask"
        mask = seg.segment(src)
        # Displays original image, mask and mask applied on original image
        seg.display()
    plt.close()


if __name__ == '__main__':
    main()
