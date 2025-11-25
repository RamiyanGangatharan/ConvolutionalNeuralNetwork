# Convolutional Neural Network from Scratch in Java
*A step-by-step tutorial series (Java implementation)*

This repository accompanies the YouTube series: **“Convolutional Neural Network from Scratch”** 
(playlist [link](https://youtu.be/3MMonOWGe0M?list=PLpcNcOt2pg8k_YsrMjSwVdy3GX-rc_ZgN)).  
The goal: build a basic CNN in Java, understand each component’s workings, and apply it to simple image classification tasks.

---

# Acknowledgments

I would like to formally acknowledge the creator of the “Convolutional Neural Network from Scratch in Java” tutorial 
series on YouTube. Their clear explanations and structured walkthrough greatly supported the development of this project 
and provided foundational guidance on implementing convolutional neural networks in Java. I am grateful for the time, effort, 
and expertise they shared in producing the series, which served as an essential learning resource throughout this work.

## Series Overview
The series is structured so you’ll learn and implement each building block of a CNN:
- Episode 0: Introduction & project setup
- Episode 1: Reading MNIST data
- Episode 2: Abstract Layer Class
- Episode 3: Fully Connected Layer
- Episode 4: Pooling layers (max-pooling)
- Episode 5: Convolution Layer Forward Pass
- Episode 6: Convolution Layer Backpropagation
- Episode 7: Training on a small dataset, and testing it


## Learning Goals
- How convolution filters extract spatial patterns from images
- How activation functions introduce non-linearity
- How pooling layers reduce spatial dimensions while preserving essential features
- How to design and implement fully connected layers
- How to perform forward and backward propagation in convolutional architectures
- How to train a neural network end-to-end using gradient descent
- How to evaluate performance on a standard dataset such as MNIST

---

## Getting Started
### Prerequisites
- Java 8 or higher
- Maven or Gradle (or simply `javac`/`java`, depending on setup)
- Basic knowledge of Java and neural-networks

## Dataset Setup
- Use a small sample dataset (I am using MNIST)
- Place image files in `data/` (or update path accordingly)
- Ensure images are greyscale or RGB as expected by the code
- 