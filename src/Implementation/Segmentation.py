import cv2
import numpy as np
import os

def PreProcessBeforeSegmentation(imagePATH):
    image = cv2.imread(imagePATH)
    if image is None:
        print(f"Error: Unable to read image at {imagePATH}")
        return None
    
    grayscaleImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurredImage = cv2.GaussianBlur(grayscaleImage, (5, 5), 0)
    _, binaryImage = cv2.threshold(blurredImage, 150, 255, cv2.THRESH_BINARY_INV)
    return binaryImage

def SegmentLines(image, lineOutputDirectory = r"Data/SegmentedLines"):
    if not os.path.exists(lineOutputDirectory):
        os.makedirs(lineOutputDirectory, exist_ok=True)

    kernel = np.ones((5, 100), np.uint8)
    dilated = cv2.dilate(image, kernel, iterations=1)
    contours, _ = cv2.findContours(dilated.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    sortedContours = sorted(contours, key=lambda ctr: cv2.boundingRect(ctr)[1])  

    return sortedContours, dilated

def SegmentWords(image, lineContours, wordOutputDirectory=r"Data/SegmentedWords"):
    if not os.path.exists(wordOutputDirectory):
        os.makedirs(wordOutputDirectory, exist_ok=True)

    kernel = np.ones((5, 20), np.uint8)
    dilated = cv2.dilate(image, kernel, iterations=1)
    wordContours = []

    for line in lineContours:
        x, y, w, h = cv2.boundingRect(line)
        roiLine = dilated[y:y+h, x:x+w]

        contours, _ = cv2.findContours(roiLine, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        sortedContours = sorted(contours, key=lambda ctr: cv2.boundingRect(ctr)[0])  

        for word in sortedContours:
            if cv2.contourArea(word) < 100:
                continue

            x2, y2, w2, h2 = cv2.boundingRect(word)
            wordContours.append([x+x2, y+y2, w2, h2])
            wordImage = image[y+y2:y+y2+h2, x+x2:x+x2+w2]
            wordPath = os.path.join(wordOutputDirectory, f"word_{len(wordContours)}.png")
            cv2.imwrite(wordPath, wordImage)

    return wordContours

def processAndSave(imagePATH, lineOutputDirectory, wordOutputDirectory):
    enhancedImage = PreProcessBeforeSegmentation(imagePATH)
    if enhancedImage is None:
        return
    cv2.imwrite("Data/PreprocessedImage.png", PreProcessBeforeSegmentation(imagePATH))
    lineContours, _ = SegmentLines(enhancedImage, lineOutputDirectory)
    SegmentWords(enhancedImage, lineContours, wordOutputDirectory)

if __name__ == "__main__":
    imagePATH = r"images.png"
    lineOutputDirectory = f"Data/SegmentedImages/{imagePATH.split(".")[0]}/SegmentedLines"
    wordOutputDirectory = f"Data/SegmentedImages/{imagePATH.split(".")[0]}/SegmentedWords"
    processAndSave(imagePATH, lineOutputDirectory, wordOutputDirectory)
