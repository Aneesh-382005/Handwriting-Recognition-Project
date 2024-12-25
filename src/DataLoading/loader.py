import os
import pandas as pd

def loadLabels(labelsFile, wordsPATH):
    labels = []

    with open(labelsFile, 'r') as file:
        for line in file:
            if line.startswith('#'):  #Ignore the commented lines
                continue

            lineParts = line.strip().split()
            if len(lineParts) < 8:
                print(f"Skipping line due to insufficient fields: '{line.strip()}'")
                continue

            wordID, segmentationStatus = lineParts[0], lineParts[1]

            try:
                grayLevel = int(lineParts[2]) if len(lineParts) > 2 else None
                x = int(lineParts[3]) if len(lineParts) > 3 else None
                y = int(lineParts[4]) if len(lineParts) > 4 else None
                w = int(lineParts[5]) if len(lineParts) > 5 else None
                h = int(lineParts[6]) if len(lineParts) > 6 else None
                grammaticalTag = lineParts[7] if len(lineParts) > 7 else None
                
                word = " ".join(lineParts[8:]) if len(lineParts) > 8 else None

                folder = wordID.split('-')[0]
                subFolder = folder + '-' + wordID.split('-')[1]
                image = f"{wordID}.png"
                imagePath = os.path.join(wordsPATH, folder, subFolder, image)

                if os.path.exists(imagePath):
                    labels.append({
                        'wordID': wordID,
                        'segmentationStatus': segmentationStatus,
                        'grayLevel': grayLevel,
                        'boundingBoxX': x,
                        'boundingBoxY': y,
                        'boundingBoxWidth': w,
                        'boundingBoxHeight': h,
                        'grammaticalTag': grammaticalTag,
                        'word': word,
                        'imagePath': imagePath
                    })
                else:
                    print(f"Image not found for imagePath {imagePath}, skipping.")
            
            except ValueError as e:
                print(f"Skipping line due to parsing error in integer fields: '{line.strip()}' - {e}")
                
    return pd.DataFrame(labels)
