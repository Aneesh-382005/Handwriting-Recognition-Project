import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNBiLSTMCTC(nn.Module):
    def __init__(self, numberofClasses, hiddenSize = 512):
        super(CNNBiLSTMCTC, self).__init__()
        
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size = 3, stride = 1, padding = 1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size = 3, stride = 1, padding = 1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = (2, 1), stride = (2, 1)),
            nn.Conv2d(128, 256, kernel_size = 3, stride = 1, padding = 1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = (2, 1), stride = (2, 1)),
        )

        self.biLSTM = nn.LSTM(128 * 8, num_layers = 2, bidirectional = True, batch_first = True, hidden_size = hiddenSize)
        self.fc = nn.Linear(hiddenSize * 2, numberofClasses)

    def forward(self, x):
        features = self.cnn(x)
        batch, channels, height, width = features.size()

        features = features.permute(0, 3, 1, 2).contiguous().view(batch, width, channels * height)
        lstmOUT, _ = self.biLSTM(features)
        logits = self.fc(lstmOUT)
        logits = F.log_softmax(logits, dim=2)
        return logits
    
