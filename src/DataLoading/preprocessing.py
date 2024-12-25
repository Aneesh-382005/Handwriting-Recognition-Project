import cv2
import numpy as np
import random
import os
import pandas as pd

def toGrayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def binarize(image):
    _, binaryImage = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binaryImage

def denoise(image):
    return cv2.medianBlur(image, 3)

def enhanceText(image):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    return cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)

def resizeAndPad(image, targetHeight = 32, targetWidth = 128):
    height, width = image.shape
    if height <= 0 or width <= 0:
        raise ValueError("Image dimensions are invalid (height or width <= 0)")

    scale = targetHeight / height
    resizedWidth = int(width * scale)
    
    deltaWidth = max(0, targetWidth - resizedWidth)
    padding = ((0, 0), (deltaWidth // 2, deltaWidth - deltaWidth // 2))
    
    paddedImage = np.pad(image, padding, mode='constant', constant_values=255)
    return paddedImage

def augmentImage(image):
    angle  = random.uniform(-5, 5)
    (height, width) = image.shape[:2]
    rotationMatrix = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1.0)
    rotatedImage = cv2.warpAffine(image, rotationMatrix, (width, height), borderValue = 255)

    contrastFactor = random.uniform(0.8, 1.2)
    augmentedImage = cv2.convertScaleAbs(rotatedImage, alpha = contrastFactor, beta = 0)
    return augmentedImage

def processImage(imagePath, targetHeight = 32, targetWidth = 128):
    image = cv2.imread(imagePath)
    if image is None:
        print(f"Error: Unable to read image at {imagePath}")
        return None 

    grayscaleImage = toGrayscale(image)
    binaryImage = binarize(grayscaleImage)
    denoisedImage = denoise(binaryImage)
    enhancedImage = enhanceText(denoisedImage)
    resizedImage = resizeAndPad(enhancedImage, targetHeight, targetWidth)
    augmentedImage = augmentImage(resizedImage)
    return augmentedImage

import os
import cv2
import pandas as pd

def savePreprocessedImages(outputDirectory, labelsCSV, processedLabelsPath, baseDir='Dataset\\archive\\iam_words\\words'):
    labels = pd.read_csv(labelsCSV)

    if not os.path.exists(outputDirectory):
        os.makedirs(outputDirectory)

    processedRecords = []
    for index, row in labels.iterrows():
        imagePath = row['imagePath']
        
        if not os.path.exists(imagePath):
            print(f"Image file not found at {imagePath}, skipping.")
            continue
        
        processedImage = processImage(imagePath)
        
        if processedImage is not None and processedImage.size != 0:
            relativePath = os.path.relpath(imagePath, baseDir)
            
            processedImagePath = os.path.join(outputDirectory, relativePath)
            processedDir = os.path.dirname(processedImagePath)
            if not os.path.exists(processedDir):
                os.makedirs(processedDir)
            
            success = cv2.imwrite(processedImagePath, processedImage)
            
            if success:
                processedRecords.append({
                    'wordID': row['wordID'],
                    'segmentationStatus': row['segmentationStatus'],
                    'grayLevel': row['grayLevel'],
                    'boundingBoxX': row['boundingBoxX'],
                    'boundingBoxY': row['boundingBoxY'],
                    'boundingBoxWidth': row['boundingBoxWidth'],
                    'boundingBoxHeight': row['boundingBoxHeight'],
                    'grammaticalTag': row['grammaticalTag'],
                    'word': row['word'],
                    'imagePath': processedImagePath
                })
            else:
                print(f"Error saving image: {processedImagePath}")
        else:
            print(f"Skipping image due to invalid data or empty result: {imagePath}")

    processedLabels = pd.DataFrame(processedRecords)
    processedLabels.to_csv(os.path.join(processedLabelsPath, 'LabelsForPreprocessedImages.csv'), index=False)

