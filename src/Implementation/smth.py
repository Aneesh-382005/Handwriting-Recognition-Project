from ultralytics import YOLO
import cv2
import os
from matplotlib import pyplot as plt

# Load trained model
model_path = r"models/HandPrint6/weights/best.pt"
model = YOLO(model_path)

# Path to test images
test_dir = r'Datasets/Hand_print_FRCNN.v2i.yolov8/test/images'
test_images = [os.path.join(test_dir, img) for img in os.listdir(test_dir)]

# Iterate through test images
for img_path in test_images:
    image = cv2.imread(img_path)
    results = model.predict(img_path)
    
    # Display results
    for result in results:
        annotated_image = result.plot()
        plt.imshow(annotated_image)
        plt.axis('off') 
        plt.show()  
