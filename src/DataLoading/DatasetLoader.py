import os
import pandas as pd
from PIL import Image, UnidentifiedImageError
from torch.utils.data import Dataset, random_split
from torchvision import transforms
import torch

class IAMWordDataset(Dataset):
    def __init__(self, csvFile, transform = None, statusFilter = None):
        """
        Arguments:
        csvFile (string): Path to the csv file with annotations.
        transform (callable, optional): Optional transform to be applied on a sample.
        statusFilter (string, optional): Optional filter to only include samples with a specific status. ('ok', 'err', or None for both)
        """

        self.data = pd.read_csv(csvFile)
        self.transform = transform if transform else transforms.ToTensor()
        self.statusFilter = statusFilter

        if self.statusFilter:
            self.data = self.data[self.data['segmentationStatus'] == self.statusFilter]
            self.data.reset_index(drop = True, inplace = True)


    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()

        imgPath = self.data.iloc[index]["imagePath"]
        groundTruth = self.data.iloc[index]["word"]
        try:
            image = Image.open(imgPath).convert('RGB')
        except (FileNotFoundError, UnidentifiedImageError) as e:
            raise RuntimeError(f"Error loading image {imgPath}: {e}")
        
        if self.transform:
            image = self.transform(image)

        return image, groundTruth
        