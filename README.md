# 🐱🐶 Cat vs Dog Image Classifier (CNN using TensorFlow)

This project is a **deep-learning based image classification system** that detects whether an image contains a **cat** or a **dog**.  
It uses a **Convolutional Neural Network (CNN)** trained on a labeled dataset and provides predictions along with **confidence percentage**.

---

## 🚀 Features

- ✔️ Image upload & prediction  
- ✔️ CNN-based classification model  
- ✔️ High accuracy with probability output  
- ✔️ Streamlit UI  
- ✔️ Docker-ready project  

---

## 🧠 Why Use CNN for Image Classification?

CNNs (**Convolutional Neural Networks**) are the industry-standard for computer vision because:

### ✔ Understand Spatial Features  
They automatically detect edges, textures, shapes, and patterns.

### ✔ Require Less Preprocessing  
CNNs **learn features automatically**, no manual feature engineering needed.

### ✔ Scale Well with Data  
More training data → better learning.

### ✔ Used in Real-World Systems  
- Object Detection (YOLO, SSD)  
- Face Recognition  
- Medical Imaging  
- Autonomous Vehicles  

---

## 🧩 CNN Architecture Used

**Input → Conv2D → ReLU → MaxPooling → Conv2D → ReLU → MaxPooling → Flatten → Dense → Dropout → Dense (Softmax)**

### Layer Description

| Layer               | Purpose                                   |
|---------------------|-------------------------------------------|
| **Conv2D**          | Extracts pattern features (edges/shapes)   |
| **ReLU Activation** | Adds non-linearity                         |
| **MaxPooling**      | Reduces dimensionality & prevents overfit  |
| **Flatten**         | Converts feature maps to 1D                |
| **Dense Layer**     | Learns final classification patterns        |
| **Dropout**         | Reduces overfitting                        |
| **Softmax Output**  | Generates probability scores               |

---

## 📉 Limitations of CNNs

Even powerful CNNs have some drawbacks:

- ❌ Requires large datasets  
- ❌ High computational cost (CPU is slow; GPU recommended)  
- ❌ Sensitive to image noise, blur, and lighting  
- ❌ Not explainable (black-box nature)  

---

## 📦 Project Structure

```text
catdog/
│── data/               # Training dataset (excluded from GitHub)
│── models/
│    └── catdog_cnn.h5  # Saved model
│── src/
│    ├── train.py       # Training script
│    ├── app.py         # Streamlit app
│── requirements.txt
│── Dockerfile
│── docker-compose.yml
└── README.md
