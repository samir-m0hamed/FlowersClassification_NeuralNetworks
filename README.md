# Flower Classification with Non Linear Neural Networks & Hyperparameter Tuning

![TensorFlow](https://img.shields.io/badge/TensorFlow-2.0+-FF6F00?logo=tensorflow)
![WandB](https://img.shields.io/badge/Weights_&_Biases-Sweep-FFBE00?logo=weightsandbiases)
![Python](https://img.shields.io/badge/Python-3.8+-3776AB?logo=python)

A deep learning project exploring the performance of **Linear Neural Networks (Dense Layers)** on image classification tasks. This repository demonstrates a professional workflow for model training, evaluation, and **Hyperparameter Tuning** using Weights & Biases (W&B) Sweeps.

## 📌 Project Overview
The goal of this project is to classify images of flowers (Daisy, Dandelion, Rose, Sunflower, Tulip) using a baseline fully connected network. Instead of guessing parameters, the project utilizes **Grid Search** to scientifically determine the optimal model configuration.

### Key Features
- **Data Pipeline:** Efficient image loading, resizing, and normalization (0-1 scaling) using `tf.keras.utils`.
- **Model Architecture:** A configurable Linear NN (Flatten -> Dense -> Output) to test baseline performance without Convolutions.
- **Experiment Tracking:** Real-time logging of Training/Validation Loss and Accuracy using `wandb`.
- **Automated Tuning:** A W&B Sweep that tests **32 different combinations** of hyperparameters.

## 🎛️ Hyperparameter Tuning (Sweep)
The project implements a **Grid Search** strategy to optimize the following parameters:

| Parameter | Values Tested | Description |
| :--- | :--- | :--- |
| **Batch Size** | `[8, 16]` | Balances gradient stability vs. speed. |
| **Learning Rate** | `[0.001, 0.0001]` | Controls the step size of the optimizer. |
| **Hidden Nodes** | `[64, 128]` | Determines the capacity of the dense layer. |
| **Image Size** | `[16, 224]` | Tests low vs. high-resolution input impact. |
| **Epochs** | `[5, 10]` | Monitors convergence speed. |

## 📊 Results
The results of the hyperparameter sweep are stored in `Hyperparameter Tuning Results.csv`. The integration with W&B allows for visualizing the correlation between:

## 📈 Future Improvements
- Implement Convolutional Neural Networks (CNNs) to capture spatial hierarchies.

- Add Data Augmentation to prevent overfitting on the small dataset.

- Deploy the best model using TF-Lite for mobile inference.

## 📝 License
This project is open-source and available under the MIT License.
