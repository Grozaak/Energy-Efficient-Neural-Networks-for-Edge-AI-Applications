Title : Energy-Efficient Neural Networks for Edge AI Applications

Project Overview:

This project focuses on designing and optimizing neural networks for energy-efficient deployment on edge devices. Traditional deep learning 
models consume high computational power and energy, making them unsuitable for resource-constrained environments such as IoT devices, mobile phones, and embedded systems.
The project demonstrates how model optimization techniques, specifically quantization, can significantly reduce model size and energy consumption while maintaining acceptable accuracy.

 Objectives:

->Build a baseline neural network for image classification
->Optimize the model for edge deployment
->Reduce memory usage and computational cost
->Deploy the optimized model using TensorFlow Lite
->Compare baseline and optimized models

Key Concepts Used:

Neural Networks
Edge AI
Model Quantization
TensorFlow Lite
Energy-Efficient AI

🗂️ Dataset:

MNIST Dataset
Handwritten digits (0–9)
60,000 training images
10,000 testing images
Image size: 28×28 pixels
The dataset is automatically loaded using TensorFlow.

 Technologies Used:

Python
TensorFlow & Keras
TensorFlow Lite
NumPy
Matplotlib
Vs code

 Methodology:

1)Load and preprocess the MNIST dataset
2)Train a baseline neural network using TensorFlow
3)Evaluate baseline accuracy and model size
4)Apply post-training quantization
5)Convert the model to TensorFlow Lite
6)Compare performance and efficiency

 Results:

| Model Type              | Model Size | Accuracy       | Energy Usage |
| ----------------------- | ---------- | -------------- | ------------ |
| Baseline Neural Network | Large      | High           | High         |
| Quantized Edge Model    | Small      | Slightly Lower | Low          |


The optimized model demonstrates significant reduction in size and computational requirements, making it suitable for edge AI applications.

Conclusion:

This project proves that neural networks can be effectively optimized for edge environments using quantization techniques. The optimized model achieves 
reduced memory usage and energy consumption while maintaining competitive accuracy, making it ideal for deployment on resource-constrained devices.
