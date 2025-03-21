# Handwriting Recognition Project

This project uses the IAM Handwriting Word Database to train and test a handwriting recognition model. The IAM dataset contains a large collection of handwritten English words and their corresponding transcriptions, which is useful for building and evaluating handwriting recognition systems.

## Dataset

The dataset used in this project is the [IAM Handwriting Word Database](https://www.kaggle.com/datasets/nibinv23/iam-handwriting-word-database/code), available on Kaggle. It includes images of handwritten words and their ground-truth transcriptions.

### How to Download the Dataset

To use this dataset, follow these steps:

1. Visit the [IAM Handwriting Word Database Kaggle page](https://www.kaggle.com/datasets/nibinv23/iam-handwriting-word-database/code).
2. Sign in to Kaggle (or create an account if you don't have one).
3. Download the dataset by clicking the "Download" button on the Kaggle page.
4. Extract the dataset's contents to the appropriate directory in your project.

The expected folder structure after extracting the dataset should look like this:

## Project Directory Structure

Dataset  
&nbsp;&nbsp;&nbsp;&nbsp;├── archive  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;└── iam_words  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;└── words  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├── a01  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├── a01-000u  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;└── a01-001u  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;└── a02




## Features

- **Image Preprocessing:** Utilizes OpenCV to preprocess handwritten images, enhancing recognition accuracy.
- **C++ and Python Integration:** Employs pybind11 to bind C++ code with Python, ensuring high performance and flexibility.
- **Modular Design:** Organized codebase with distinct modules for different stages of handwriting recognition.


## Credits

This project utilizes code from [Arthur Flor's Handwritten Text Segmentation repository](https://github.com/arthurflor23/handwritten-text-segmentation). The [/src/Implementation/ImageProcessing/cpp](https://github.com/Aneesh-382005/Handwriting-Recognition-Project/tree/main/src/Implementation/ImageProcessing/cpp) folder has been adapted with changes to:

- [**main.cpp**](https://github.com/Aneesh-382005/Handwriting-Recognition-Project/blob/main/src/Implementation/ImageProcessing/cpp/main.cpp): Minor modifications and component moved to **ImageProcessing.cpp**
- [**bindings.cpp**](https://github.com/Aneesh-382005/Handwriting-Recognition-Project/blob/main/src/Implementation/ImageProcessing/cpp/bindings.cpp): Bindings to facilitate integration between C++ components and Python.


These enhancements aim to optimize performance and provide greater flexibility for users engaging with the handwriting recognition system.

