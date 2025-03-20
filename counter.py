import pandas as pd

data = pd.read_csv('Data\LabelsForPreprocessedImages.csv')
words = data.loc[data['segmentationStatus'] == 'ok', 'word'].tolist()

uniqueCharacters = set()
for word in words:
    for character in word:
        uniqueCharacters.add(character)

uniqueCharacters = sorted(uniqueCharacters)
chartoIndex = {character: index for index, character in enumerate(uniqueCharacters)}
indexToChar = {index: character for character, index in chartoIndex.items()}

print("Unique characters: ", uniqueCharacters)
print("Character to index: ", chartoIndex)
print(len(uniqueCharacters))
