'''data = pd.read_csv('Data/LabelsForPreprocessedImages.csv')
words = data['word'].tolist()
uniqueCharacters = sorted(set(''.join(words)))
chartoIndex = {character: index + 1 for index, character in enumerate(uniqueCharacters)}
numberofClasses = len(uniqueCharacters) + 1
alphabet = [''] + uniqueCharacters'''


import torch
from PIL import Image
from torchvision import transforms
from src.Implementation.Models.CNN_BiLSTM_CTC import CNNBiLSTMCTC
import os
from datetime import datetime

device = 'cuda'

def load_model(model_path, numberofClasses, device):
    model = CNNBiLSTMCTC(numberofClasses).to(device)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model

def preprocess_image(image_path, transform):
    image = Image.open(image_path).convert('L')
    return transform(image).unsqueeze(0)

def decode_ctc_output(output, alphabet):
    return ''.join([alphabet[i] for i in torch.argmax(output.squeeze(0), dim=1) if i != 0])

transform = transforms.Compose([
    transforms.Resize((32, 128)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)),
])

alphabet = ['', ' ', '!', '"', '#', "'", '(', ')', '*', ',', '-', '.', '/', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', ':', ';', '?', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
numberofClasses = len(alphabet)

model = load_model('checkpoints\\CNN_BiLSTM_CTC\\modelEpoch30.pth', numberofClasses, device)

path = 'outputText\\words'
predictedLine = []

t1 = datetime.now()

with torch.no_grad():
    for word in os.listdir(path):
        image_path = os.path.join(path, word)
        image = preprocess_image(image_path, transform).to(device)
        output = model(image)
        predictedLine.append(decode_ctc_output(output, alphabet))

print(predictedLine)
print(datetime.now() - t1)
