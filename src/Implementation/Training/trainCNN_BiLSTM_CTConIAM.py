import torch.optim as optim
import torch.nn.functional as F
from torch.optim import lr_scheduler
import torch 
import torch.nn as nn
import torch.utils.data

import sys
import os
import pandas as pd


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

from src.Implementation.Models.CNN_BiLSTM_CTC import CNNBiLSTMCTC
from src.DataLoading.DatasetLoader import IAMWordDataset

import pandas as pd

data = pd.read_csv('Data/LabelsForPreprocessedImages.csv')
words = data['word'].tolist()

uniqueCharacters = set()
for word in words:
    for character in word:
        uniqueCharacters.add(character)

uniqueCharacters = sorted(uniqueCharacters)
chartoIndex = {character: index + 1 for index, character in enumerate(uniqueCharacters)}



def ConvertTargets(targets):
    targetIndices = []
    targetLengths = []
    for label in targets:
        indices = [chartoIndex.get(character, 0) for character in label]
        targetIndices.extend(indices)
        targetLengths.append(len(indices))
    
    targetSequence = torch.tensor(targetIndices, dtype = torch.long)
    targetLengths = torch.tensor(targetLengths, dtype = torch.long)

    return targetSequence, targetLengths


def TrainModel(model, trainLoader, numberofEpochs, optimizer, scheduler, CTCLoss, device):
    model.train()
    for epoch in range(numberofEpochs):
        epochLoss = 0.0
        for images, labels in trainLoader:
            images = images.to(device)
            targetSequence, targetLengths = ConvertTargets(labels)
            targetSequence = targetSequence.to(device)

            logits = model(images).permute(1, 0, 2)
            T, batchSize, _ = logits.size()

            inputLengths = torch.full(size = (batchSize,), fill_value = T, dtype = torch.long, device = device)

            loss = CTCLoss(logits, targetSequence, inputLengths, targetLengths)

            optimizer.zero_grad()
            loss.backward()
                
            optimizer.step()

            epochLoss += loss.item()

        scheduler.step()
        averageLoss = epochLoss / len(trainLoader)
        print(f"Epoch [{epoch + 1}/{numberofEpochs}], Loss: {averageLoss:.4f}")
        
        checkpointPath = os.path.join("Checkpoints", "CNN_BiLSTM_CTC", f'modelEpoch{epoch + 1}.pth')
        os.makedirs(os.path.dirname(checkpointPath), exist_ok = True)
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': averageLoss
        }, checkpointPath)
        
        print(f"Checkpoint saved at {checkpointPath}")
        
if __name__ == "__main__":     
    numberofClasses = len(uniqueCharacters) + 1
    batchSize = 32
    learningRate = 1e-3
    numberofEpochs = 30

    os.environ['NO_ALBUMENTATIONS_UPDATE'] = '1'

    from albumentations import Compose, RandomBrightnessContrast, Normalize, Resize, Affine
    from albumentations.pytorch import ToTensorV2

    transform = Compose([
    Resize(height=32, width=128, interpolation=1),
    Affine(
        scale=(0.9, 1.1),     # Scale changes between 90% and 110%
        translate_percent = (0.05, 0.05), # Shift by 5% in both directions
        rotate=10,           # Rotate by up to 10 degrees
        border_mode=0,      # Use constant border with value 0 (black)
        p=0.7               # Apply with 70% probability
    ),
    RandomBrightnessContrast(p=0.2),
    Normalize(mean=(0.5,), std=(0.5,)),
    ToTensorV2(),
    ])

    dataset = IAMWordDataset("Data/LabelsForPreprocessedImages.csv", transform = transform, statusFilter = 'ok')
    trainLoader = torch.utils.data.DataLoader(dataset, batch_size = batchSize, shuffle = True, num_workers = 8)



    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    model = CNNBiLSTMCTC(numberofClasses).to(device)
    optimizer = optim.Adam(model.parameters(), lr = learningRate)
    scheduler = lr_scheduler.StepLR(optimizer, step_size = 10, gamma = 0.1)
    CTCLoss = nn.CTCLoss(blank = 0)

    TrainModel(model, trainLoader, numberofEpochs, optimizer, scheduler, CTCLoss, device)

    torch.save(model.state_dict(), '')


    
