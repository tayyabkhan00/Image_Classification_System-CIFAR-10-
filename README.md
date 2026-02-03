# 🧠 CIFAR-10 Image Classification using ResNet-18

A **deep learning web application** that classifies images into 10 categories using a **ResNet-18 Convolutional Neural Network**, trained on the CIFAR-10 dataset and deployed with **Streamlit Cloud**.

This project demonstrates an **end-to-end Deep Learning workflow** — from model training to real-world deployment with a clean, interactive UI.

---

## 🚀 Live Demo
👉 **Streamlit App:**  
*(https://aqcdjkxheveuvl3sjidcvt.streamlit.app/)*

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

- **Train / Test Split:** 50,000 / 10,000  

📌 The dataset is **not uploaded to GitHub** (ignored via `.gitignore`)  
📌 It is automatically downloaded using `torchvision`

---

## 🎯 Results

| Model | Test Accuracy |
|-----|---------------|
| Basic CNN | ~70% |
| Improved CNN | ~80% |
| **ResNet-18** | **~88% ✅** |

---

## 🖥️ Web App Features

### 🔮 Image Prediction
- Upload JPG / PNG images
- Predict image class with confidence score
- Clean, card-based UI

### 🧪 CIFAR-10 Sample Testing
- Test predictions on real CIFAR-10 images
- Compare true label vs predicted label

### 📊 Class-wise Accuracy
- Interactive bar chart for each class
- Expandable accuracy table

---

## 🛠️ Tech Stack

- **Python**
- **PyTorch**
- **Torchvision**
- **Streamlit**
- **NumPy**
- **Matplotlib**
- **Pillow (PIL)**

---

## 📂 Project Structure
```
CNN_PROJECT/
│
├── deployment/               # Streamlit deployment
│ ├── app.py                  # Main Streamlit app
│ ├── model.py                # ResNet-18 architecture
│ ├── resnet_cifar10.pth      # Trained model weights
│ └── requirements.txt
│
├── model/                    # Training & experiments
│ ├── cnn_72.py
│ ├── cnn_84.py
│ └── resnet_18.py
│
├── data/                      # Dataset (ignored via .gitignore)
│
├── test_images.py             # Utility script
├── .gitignore
└── README.md
```

---

## ⚙️ Installation & Local Setup

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/your-username/your-repo-name.git
cd CNN_PROJECT
```
### 2️⃣ Create Virtual Environment (Optional)
```bash
python -m venv venv
source venv/bin/activate   # macOS / Linux
venv\Scripts\activate      # Windows
```
### 3️⃣ Install Dependencies
```bash
pip install -r deployment/requirements.txt
```
### 4️⃣ Run the Streamlit App
```bash
streamlit run deployment/app.py
```
### ☁️ Deployment (Streamlit Cloud)

- Platform: Streamlit Cloud
- App file path:
```bash
deployment/app.py
```
- Requirements file:
```bash
deployment/requirements.txt
```
The model weights (.pth) are loaded directly from the repository.

---

## 🧪 Training Summary

- Epochs: 30
- Batch Size: 128
- Data Augmentation:
   - Random Crop
   - Horizontal Flip
- Normalization: Mean = 0.5, Std = 0.5

---

## 🖥️ Training Environment & Compute Considerations

Training a deep architecture like **ResNet-18 on CIFAR-10** is computationally expensive.

### ⚠️ Important Note on Training Time
- Training the model **on CPU only** can take **8–10+ hours** depending on hardware.
- Due to this limitation, training on a local CPU is **not recommended**.

### ✅ Recommended: Google Colab (GPU)
To efficiently train the model and generate the `resnet_cifar10.pth` file, **Google Colab with GPU acceleration** was used.

**Benefits of using Colab:**
- Free GPU access (Tesla T4 / P100)
- Training completes in **~30–45 minutes**
- Faster experimentation and debugging
- Ideal for deep CNN architectures like ResNet

### 🔄 Workflow Used in This Project
1. Train ResNet-18 on **Google Colab (GPU)**
2. Save trained weights as `resnet_cifar10.pth`
3. Download the `.pth` file
4. Use the trained weights for:
   - Local inference
   - Streamlit Cloud deployment

This approach ensures **efficient training** while keeping deployment lightweight and reproducible.

---

## 📈 Evaluation Metrics

- Overall Test Accuracy
- Class-wise Accuracy
- Softmax Confidence Scores
 
---

## 💡 What This Project Demonstrates

- ✅ Deep Learning fundamentals
- ✅ CNN & ResNet architecture understanding
- ✅ PyTorch training pipeline
- ✅ Model optimization techniques
- ✅ Deployment using Streamlit
- ✅ Clean UI/UX for ML applications
- ✅ Proper Git & GitHub practices

This project is suitable for:
- Data Science portfolios
- Deep Learning internships
- ML / AI Engineer roles

---

## 🚀 Future Improvements

- Grad-CAM heatmap visualization
- Model comparison (VGG vs ResNet)
- Faster inference optimizations
- Mobile-friendly UI
- Cloud storage for model artifacts

---

## 👨‍💻 Author

Tayyab Khan<br>
B.Tech – AI & Data Science

**⭐ Support**

If you like this project, consider giving it a ⭐ on GitHub!
