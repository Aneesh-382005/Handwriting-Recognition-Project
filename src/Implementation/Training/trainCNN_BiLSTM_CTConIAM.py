import torch.optim as optim
import torch.nn.functional as F
import torch 

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))
print("yaya", (os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))))

from src.Implementation.Models.CNN_BiLSTM_CTC import CNNBiLSTMCTC
from src.DataLoading.DatasetLoader import IAMWordDataset


numberofClasses = 83
learningRate = 1e-3
numberofEpochs = 30

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = CNNBiLSTMCTC(numberofClasses).to(device)

dataset = IAMWordDataset("Data/LabelsForPreprocessedImages.csv", statusFilter = 'ok')
