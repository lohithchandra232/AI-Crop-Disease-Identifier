\# 🌱 AI Crop Disease Identifier



An AI-powered web application that detects tomato leaf diseases from images using a trained deep learning model.



\## 🚀 Features



\- Upload a tomato leaf image

\- Detect:

&#x20; - Tomato Early Blight

&#x20; - Tomato Late Blight

&#x20; - Tomato Healthy

\- Display prediction confidence

\- FastAPI backend for AI predictions

\- React/Vite frontend

\- Simple and user-friendly interface



\## 🧠 Machine Learning



The model was trained using tomato leaf images from the PlantVillage dataset.



\### Dataset



\- Total images: 4,500

\- Training split: 80%

\- Validation split: 20%



\### Classes



| Class | Images |

|---|---:|

| Tomato Early Blight | 1,000 |

| Tomato Late Blight | 1,909 |

| Tomato Healthy | 1,591 |



\### Validation Performance



\*\*Validation Accuracy: 93.67%\*\*



The trained Keras model is used by the FastAPI backend to classify uploaded images.



> Note: Model confidence is not a guarantee of correctness. Images that differ significantly from the training data may produce incorrect predictions.



\## 🏗️ Project Architecture



```text

User

&#x20; │

&#x20; ▼

React / Vite Frontend

&#x20; │

&#x20; │ Image Upload

&#x20; ▼

FastAPI Backend

&#x20; │

&#x20; ▼

TensorFlow / Keras Model

&#x20; │

&#x20; ▼

Disease Prediction

&#x20; │

&#x20; ▼

Prediction + Confidence

