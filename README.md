# 🧠 CIFAR-10 Image Classification using ResNet-18

A **deep learning web application** that classifies images into 10 categories using a **ResNet-18 Convolutional Neural Network**, trained on the CIFAR-10 dataset and deployed with **Streamlit Cloud**.

This project demonstrates an **end-to-end Deep Learning workflow** — from model training to real-world deployment with a clean, interactive UI.

---

## 🚀 Live Demo
👉 **Streamlit App:**  
*(Add your Streamlit Cloud URL here after deployment)*

---

## 📌 Project Overview

This project showcases:
- Training a **ResNet-18 model from scratch** using PyTorch
- Achieving **~88% accuracy** on the CIFAR-10 test set
- Deploying the trained model as a **production-style Streamlit web app**
- Providing **class-wise performance analysis**

### Users can:
- Upload custom images for prediction
- Test the model on real CIFAR-10 samples
- View prediction confidence scores
- Analyze class-wise accuracy

---

## 🧠 Model Architecture

- **Model:** ResNet-18  
- **Core Idea:** Residual (skip) connections to solve vanishing gradients  
- **Loss Function:** Cross Entropy Loss  
- **Optimizer:** SGD with Momentum  
- **Learning Rate Scheduler:** Cosine Annealing  

**Why ResNet?**  
Residual connections allow deeper networks to train efficiently by learning identity mappings.

---

## 📊 Dataset: CIFAR-10

- **60,000 color images (32×32)**
- **10 classes:**
