import torch
import torch.nn as nn

class CNNBiLSTMCTC(nn.Module):
    def __init__(self, numberofClasses, hiddenSize = 256):
        super(CNNBiLSTMCTC, self).__init__()
        
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size = 3, stride = 1, padding = 1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size = 3, stride = 1, padding = 1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )

        self.biLSTM = nn.LSTM(128 * 8, num_layers = 2, bidirectional = True, batch_first = True, hidden_size = hiddenSize)
        self.fc = nn.Linear(256 * 2, numberofClasses)

    def forward(self, x):
        features = self.cnn(x)
        batch, channels, height, width = features.size()
        features = features.permute(0, 3, 1, 2).contiguous().view(batch, width, channels * height)
        lstmOUT, _ = self.biLSTM(features)
        logits = self.fc(lstmOUT)
        logits.transpose
        return logits
    
