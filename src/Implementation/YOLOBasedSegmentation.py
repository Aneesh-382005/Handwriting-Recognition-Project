import os
from ultralytics import YOLO
import torch
from albumentations import Compose, Rotate, HorizontalFlip, Resize, RandomBrightnessContrast
from dotenv import load_dotenv

load_dotenv()
'''def AugumentData(image):
    transform = Compose([
        Rotate(limit=30),
        HorizontalFlip(p=0.5),
        Resize(height=640, width=640),
        RandomBrightnessContrast(p=0.2) 
    ])
    augumented = transform(image=image)
    return augumented['image']'''

def main():

    dataPATH = os.getenv('dataPATH')
    EPOCHS = int(os.getenv('EPOCHS'))
    DEVICE = os.getenv('DEVICE')
    WORKERS = int(os.getenv('WORKERS'))
    projectPATH = os.getenv('projectPATH')
    NAME = os.getenv('NAME')

    print(dataPATH)
    print(EPOCHS)
    print(DEVICE)
    print(WORKERS)
    print(projectPATH)
    print(NAME)


    EPOCHS = 30

    model = YOLO('yolov8n.yaml')
    model.train(
        data = dataPATH,
        epochs = EPOCHS,
        device = DEVICE, 
        workers = WORKERS, 
        project = projectPATH,
        name = NAME
    )

    os.makedirs('models', exist_ok=True)
    
    model.save(r'models/yolov8n.pt')



if __name__ == "__main__":
    import torch.multiprocessing
    torch.multiprocessing.set_start_method("spawn")  
    main()