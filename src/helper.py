import os
import cv2
import pandas as pd

from DataLoading.loader import loadLabels
from DataLoading.preprocessing import savePreprocessedImages

print(os.getcwd())

datasetPATH = os.path.join('Datasets', 'archive', 'iam_words')
wordsPATH = os.path.join(datasetPATH, 'words')
labelsPATH = os.path.join('Datasets', 'archive', 'words_new.txt')

print("Dataset Path:", datasetPATH)
print("Words Path:", wordsPATH)
print("Labels Path:", labelsPATH)

detailedLabels = loadLabels(labelsPATH, wordsPATH)
print(detailedLabels.head())
os.makedirs('Data', exist_ok=True)
csvPath = os.path.join('Data', 'Labels.csv')
detailedLabels.to_csv(csvPath, index=False)

outputDirectory = os.path.join('Data', 'PreprocessedImages')
savePreprocessedImages(outputDirectory, csvPath, 'Data')

print("Preprocessing Done!")