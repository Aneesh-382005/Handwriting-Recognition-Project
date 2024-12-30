import cv2
import numpy as np
import os

def PreProcessBeforeSegmentation(imagePATH):
    image = cv2.imread(imagePATH)
    if image is None:
        print(f"Error: Unable to read image at {imagePATH}")
        return None
    
    grayscaleImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binaryImage = cv2.threshold(grayscaleImage, 80, 255, cv2.THRESH_BINARY_INV)
    return binaryImage

def SegmentLines(image, lineOutputDirectory=r"Data/SegmentedLines"):
    if not os.path.exists(lineOutputDirectory):
        os.makedirs(lineOutputDirectory)

    kernel = np.ones((5, 85), np.uint8)
    dilated = cv2.dilate(image, kernel, iterations=1)
    contours, _ = cv2.findContours(dilated.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    sortedContours = sorted(contours, key=lambda ctr: cv2.boundingRect(ctr)[1])

    lineImages = []
    for idx, line in enumerate(sortedContours):
        x, y, w, h = cv2.boundingRect(line)
        lineImage = image[y:y+h, x:x+w]
        linePath = os.path.join(lineOutputDirectory, f"line_{idx + 1}.png")
        cv2.imwrite(linePath, lineImage)
        lineImages.append(lineImage)

    return sortedContours, lineImages

def SegmentWords(image, lineContours, wordOutputDirectory=r"Data/SegmentedWords"):
    if not os.path.exists(wordOutputDirectory):
        os.makedirs(wordOutputDirectory)

    kernel = np.ones((3, 15), np.uint8)
    dilated = cv2.dilate(image, kernel, iterations=1)
    wordContours = []
    wordIdx = 0

    for lineIdx, line in enumerate(lineContours):
        x, y, w, h = cv2.boundingRect(line)
        roiLine = dilated[y:y+h, x:x+w]

        contours, _ = cv2.findContours(roiLine, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        sortedContours = sorted(contours, key=lambda ctr: cv2.boundingRect(ctr)[0])

        for word in sortedContours:
            if cv2.contourArea(word) < 50:
                continue

            x2, y2, w2, h2 = cv2.boundingRect(word)
            wordContours.append([x+x2, y+y2, w2, h2])
            wordImage = image[y+y2:y+y2+h2, x+x2:x+x2+w2]
            wordPath = os.path.join(wordOutputDirectory, f"line_{lineIdx + 1}_word_{wordIdx + 1}.png")
            cv2.imwrite(wordPath, wordImage)
            wordIdx += 1

    return wordContours

def processAndSave(imagePATH, lineOutputDirectory, wordOutputDirectory):
    enhancedImage = PreProcessBeforeSegmentation(imagePATH)
    if enhancedImage is None:
        return

    lineContours, _ = SegmentLines(enhancedImage, lineOutputDirectory)
    SegmentWords(enhancedImage, lineContours, wordOutputDirectory)

if __name__ == "__main__":
    imagePATH = r"smth.png"
    lineOutputDirectory = r"Data/SegmentedLines"
    wordOutputDirectory = r"Data/SegmentedWords"
    processAndSave(imagePATH, lineOutputDirectory, wordOutputDirectory)
