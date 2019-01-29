'''
    Image Processing - Assignment 2
    Content Based Image Retrieval

    @author
    Student: Osman Araz
    Student NO: 16011020
    Delivery Date: 12.12.2018
'''

'''
    ATTENTION: Make sure that Python has installed on your computer. To install Python: 
    https://www.python.org/downloads/

    Also, you need some Python libraries to read and store images. Please open up your 
    terminal/shell screen and execute the commands below:
    
    pip install numpy
    pip install matplotlib
    pip install Pillow
'''

import os
import sys
import math
import platform
import numpy as np
import matplotlib.image as mpimg

class Image:
    '''
    A class which stores the datas of an image.
    '''
    def __init__(self, name, hueHistogram, LBPHistogram):
        self.name = name
        self.hueHistogram = hueHistogram
        self.LBPHistogram = LBPHistogram

    def setHueAndLBPDiffs(self, otherImage):
        self.hueDiff = compareHistograms(self.hueHistogram, otherImage.hueHistogram)
        self.LBPDiff = compareHistograms(self.LBPHistogram, otherImage.LBPHistogram)
        self.hueAndLBPDiff = self.hueDiff + self.LBPDiff

#########

def detectPatterns():
    '''
    A function which detects the LBP patterns which will be considered while 
    comparing the LBP histograms.
    '''
    isPattern = [False]*256
    isPattern[0] = True
    isPattern[255] = True

    for i in range(1, 8):
        num = (2**i)-1  # 1, 3, 7... 

        for _ in range(8-i):
            isPattern[num] = True
            isPattern[255-num] = True # 1 and 0 bits are exchanging
            num <<= 1   # shifting the number to the left for (8-i) times

    return isPattern

#########

def computeHueAndLuminance(r, g, b):
    '''
    A function which computes the hue and luminance values from a given RGB value.
    '''
    luminance = 0.299*r + 0.587*g + 0.114*b

    if (r == g and g == b):
        hue = 0
    else:
        hue = math.acos((2*r-g-b) / (2*math.sqrt((r-g)*(r-g)+(r-b)*(g-b))))*(180/math.pi)

        if (b > g):
            hue = 360-hue

    return int(hue), int(luminance)

#########

def getHueAndLuminanceValues(imageMatrix, height, width):
    '''
    A function which generates the hue and luminance matrixes from a given RGB matrix.
    '''
    hueValues, luminanceValues = [None]*height, [None]*height

    for i in range(height):
        hueValues[i], luminanceValues[i] = [None]*width, [None]*width

        for j in range(width):
            hueValues[i][j], luminanceValues[i][j] = computeHueAndLuminance(int(imageMatrix[i][j][0]), int(imageMatrix[i][j][1]), int(imageMatrix[i][j][2]))

    return hueValues, luminanceValues

#########

def getHueHistogram(hueValues, height, width):
    '''
    A function which calculates the HUE histograms of an image.
    '''
    hueHistogram = [0]*360

    for i in range(height):
        for j in range(width):
            hueHistogram[hueValues[i][j]] += 1

    for i in range(360):
        hueHistogram[i] /= height*width # normalizing the histogram

    return hueHistogram

#########

def getLBPValueSequential(luminanceValues, x, y):
    '''
    A function which calculates the LBP value for a given pixel. (Sequential Traversing)
    '''
    value = 0
    k = 7

    for i in range(x-1, x+2):
        for j in range(y-1, y+2):
            if luminanceValues[i][j] > luminanceValues[x][y]:
                value += (1 << k)

            if (i != x or j != y):
                k -= 1

    return value

#########

def getLBPValueClockwise(luminanceValues, x, y):
    '''
    A function which calculates the LBP value for a given pixel. (Clockwise Traversing)
    '''
    value = 0
    
    if luminanceValues[x-1][y+1] > luminanceValues[x][y]:
        value += 1
    
    if luminanceValues[x][y+1] > luminanceValues[x][y]:
        value += 1 << 1

    if luminanceValues[x+1][y+1] > luminanceValues[x][y]:
        value += 1 << 2

    if luminanceValues[x+1][y] > luminanceValues[x][y]:
        value += 1 << 3

    if luminanceValues[x+1][y-1] > luminanceValues[x][y]:
        value += 1 << 4

    if luminanceValues[x][y-1] > luminanceValues[x][y]:
        value += 1 << 5

    if luminanceValues[x-1][y-1] > luminanceValues[x][y]:
        value += 1 << 6

    if luminanceValues[x-1][y] > luminanceValues[x][y]:
        value += 1 << 7

    return value

#########

def getLBPMatrix(luminanceValues, height, width):
    '''
    A function which generates the LBP matrix of an image from luminance values.
    '''
    LBPMatrix = [None]*height

    for i in range(1, height-1):
        LBPMatrix[i] = [None]*width

        for j in range(1, width-1):
            LBPMatrix[i][j] = getLBPValueClockwise(luminanceValues, i, j)

    return LBPMatrix

#########

def getLBPHistogram(LBPMatrix, height, width, isPattern):
    '''
    A function which calculates the LBP histograms of an image.
    '''
    LBPHistogram = [0]*257  

    for i in range(1, height-1):
        for j in range(1, width-1):
            if (isPattern[LBPMatrix[i][j]] == True): # only patterns will be counted
                LBPHistogram[LBPMatrix[i][j]] += 1
            else: # any LBP value which is not a pattern will be considered same
                LBPHistogram[256] += 1  

    for i in range(257):
        LBPHistogram[i] /= width*height # normalizing the histogram

    return LBPHistogram

#########

def getHistogramsOfImage(imageFileName):
    '''
    A function which reads RGB values of an image and then returns with its calculated histograms.
    '''
    imageMatrix = mpimg.imread(imageFileName)   # reading the RGB values to a matrix

    height = len(imageMatrix)
    width = len(imageMatrix[0])

    hueValues, luminanceValues = getHueAndLuminanceValues(imageMatrix, height, width)
    hueHistogram = getHueHistogram(hueValues, height, width)

    LBPMatrix = getLBPMatrix(luminanceValues, height, width)
    LBPHistogram = getLBPHistogram(LBPMatrix, height, width, isPattern)

    return Image(imageFileName, hueHistogram, LBPHistogram)

#########

def getHistogramsOfComparisonImages(comparisonImagesPath):
    '''
    A function which gets images within a given directory and then sets their histograms.
    '''
    pathsOfImages = os.listdir(comparisonImagesPath) # getting the list of images within the given directory

    comparisonImages = []

    for imagePath in pathsOfImages:
        print("\tgetting", imagePath, "...")
        comparisonImages.append(getHistogramsOfImage(comparisonImagesPath + "/" + imagePath))
    
    return comparisonImages # returning the gotten images

#########

def compareHistograms(histogram1, histogram2):
    '''
    A function which returns the difference of HUE or LBP histograms of two images. 
    '''
    diff = 0

    for i in range(len(histogram1)):
        diff += abs(histogram1[i]-histogram2[i])

    return diff

#########

def setDifferencesOfImages(mainImage, comparisonImages):
    '''
    A function which sets the HUE and LBP differences between a given image and a collection of images.
    '''
    for image in comparisonImages:
        image.setHueAndLBPDiffs(mainImage)

#########

def printResults(image, comparisonImages):
    '''
    A function which prints the names of the most similar images to a given image.
    '''
    printBanner()

    # sorting the images according to the HUE differences
    comparisonImages.sort(key=lambda image: image.hueDiff)

    print(f"\tMost similar images to {image.name} according to the HUE differences:\n")

    for i in range(min(5, len(comparisonImages))):
        print("\t", comparisonImages[i].name, " (", comparisonImages[i].hueDiff, ")", sep = "")

    # sorting the images according to the LBP differences
    comparisonImages.sort(key=lambda image: image.LBPDiff)

    print(f"\n\tMost similar images to {image.name} according to the LBP differences:\n")

    for i in range(min(5, len(comparisonImages))):
        print("\t", comparisonImages[i].name, " (", comparisonImages[i].LBPDiff, ")", sep = "")

    # sorting the images according to the HUE and LBP differences
    comparisonImages.sort(key=lambda image: image.hueAndLBPDiff)

    print(f"\n\tMost similar images to {image.name} according to the HUE and LBP differences:\n")

    for i in range(min(5, len(comparisonImages))):
        print("\t", comparisonImages[i].name, " (", comparisonImages[i].hueAndLBPDiff, ")", sep = "")

#########

def clearScreen():
    '''
    A function which clears the terminal/shell screen according to the operating system.
    '''
    if platform.system() == "Windows":
        os.system("cls")
    else:                  # Linux & Mac OS
        os.system("clear")

#########

def printBanner():
    '''
    A function which prints the banner of the program.
    '''
    clearScreen()
    print("\n\t#######################################")
    print("\t################# CBIR ################")
    print("\t#######################################\n")

#########

def main():
    printBanner()

    fileExists = False

    while fileExists == False:
        comparisonImagesPath = input("\tEnter the path of directory of data images:\n\t(the directory should consists of images only): ")
        fileExists = os.path.exists(comparisonImagesPath)   # checking if the path is valid or not

        if (fileExists == False):
            print("\n\tWrong directory path, try again:")

    print()
    comparisonImages = getHistogramsOfComparisonImages(comparisonImagesPath)

    cont = 'Y'  #  a variable to get choice from user (yes or no)

    while (cont.upper() == 'Y'):
        printBanner()

        fileExists = False

        while fileExists == False:
            imagePath = input("\tEnter the path of input image: ")
            fileExists = os.path.exists(imagePath)  # checking if the path is valid or not

            if (fileExists == False):
                print("\n\tWrong image path, try again:")

        image = getHistogramsOfImage(imagePath)

        setDifferencesOfImages(image, comparisonImages)

        printResults(image, comparisonImages)

        cont = input("\n\tTry for another image (Y/N) ?: ") 

#########

isPattern = detectPatterns()    # detecting the LBP patterns

if __name__ == "__main__":
    main()