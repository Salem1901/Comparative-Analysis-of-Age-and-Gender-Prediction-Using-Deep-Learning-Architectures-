# Comparative-Analysis-of-Age-and-Gender-Prediction-Using-Deep-Learning-Architectures-
This repository contains the implementation of a deep learning-based approach for predicting gender (classification) and age (regression) from panoramic dental X-ray images. The project compares various state-of-the-art models, including CNN, Vision Transformer, ResNet, MobileNet, DenseNet, and Vision-Language Models (VLMs) like Moondream2. It also explores data preprocessing techniques and inference time for single-image predictions.
# Key Features
Comprehensive Model Comparison: Includes CNN, VGG16, VGG19, ResNet50, ResNet101, ResNet152, MobileNet, DenseNet121, DenseNet169, Vision Transformer, and Moondream2.
Dual-Task Learning: Predicts gender (classification) and age (regression) using separate models.
Class Imbalance Mitigation: Utilizes Random Over Sampler to address dataset imbalance.
Inference Time Analysis: Benchmarks the time taken by each model for single-image predictions.
Exploration of Vision-Language Models: Evaluates VLMs like Moondream2 for medical imaging tasks.
Customizable Code: Modular implementation for easy adaptation and experimentation.

# Requirements
To run the code, ensure you have the following installed:

Python 3.8 or above
TensorFlow 2.10+
Keras 2.9+
NumPy 1.21+
Pandas 1.3+
Scikit-learn 1.0+
Matplotlib 3.4+
albumentations 1.2.0
Pillow 8.4+

# Install the dependencies using:
pip install -r requirements.txt  

# How to Run
Clone the Repository
git clone https://github.com/<your-username>/Comparative-Analysis-of-Age-and-Gender-Prediction-Using-Deep-Learning-Architectures-.git  
cd Comparative-Analysis-of-Age-and-Gender-Prediction-Using-Deep-Learning-Architectures-  

Place the dataset in the appropriate directory or folder as required by the notebooks. Ensure your dataset is formatted and structured as expected in each experiment.
xplore the Notebooks
The repository contains the following Jupyter notebooks for training and experimenting with different models:

Experiment_CNN_DENSENET121.ipynb
Experiment_CNN_DENSENET169.ipynb
Experiment_CNN_MOBILENET.ipynb
Experiment_CNN_RESNET50.ipynb
Experiment_CNN_RESNET101.ipynb
Experiment_CNN_RESNET152.ipynb
Experiment_CNN_ROS.ipynb
Experiment_CNN_VGGT16.ipynb
Experiment_CNN_VGGT19.ipynb
Experiment_Moondream2.ipynb


