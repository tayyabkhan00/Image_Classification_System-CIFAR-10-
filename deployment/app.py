import streamlit as st
import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from model import ResNet18
import os
import random
import torchvision
import time

if "sample_idx" not in st.session_state:
    st.session_state.sample_idx = random.randint(0, 9999)

# -------------------------
# -------------------------
@st.cache_resource
def load_cifar_samples():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
    ])
    dataset = torchvision.datasets.CIFAR10(
        root="./data", train=False, download=True, transform=transform
    )
    return dataset
# -----------------------------
# 

def animated_confidence_bar(confidence, duration=0.8):
    """
    confidence: float (0–100)
    duration: total animation time in seconds
    """
    progress_placeholder = st.empty()
    steps = int(confallow := max(1, int(confidence)))  # safe steps
    sleep_time = duration / steps

    progress = 0
    for i in range(steps):
        progress += 1
        progress_placeholder.progress(progress / 100)
        time.sleep(sleep_time)

    st.caption(f"Confidence: {confidence:.2f}%")

# -----------------------------
# Page config
# -----------------------------
st.set_page_config(page_title="CIFAR-10 Image Classifier", layout="centered")
# -----------------------------
# -----------------------------
st.markdown("""
<style>
/* Main background */
.stApp {
    background-color: #0f172a;
    color: #e5e7eb;
}

/* Titles */
h1, h2, h3 {
    color: #f8fafc;
}

/* Card styling */
.card {
    background: #020617;
    padding: 20px;
    border-radius: 16px;
    box-shadow: 0 10px 30px rgba(0,0,0,0.4);
    margin-bottom: 20px;
}

/* Prediction text */
.pred {
    font-size: 22px;
    font-weight: 600;
    color: #38bdf8;
}

/* Confidence bar */
.conf-bar {
    background-color: #1e293b;
    border-radius: 10px;
    overflow: hidden;
    margin-top: 10px;
}

.conf-fill {
    height: 14px;
    background: linear-gradient(90deg, #22d3ee, #38bdf8);
}

/* Footer */
.footer {
    text-align: center;
    color: #94a3b8;
    margin-top: 50px;
    font-size: 14px;
}
</style>
""", unsafe_allow_html=True)

# -----------------------------
# -----------------------------
st.markdown("""
<div class="card">
<h1>🧠 CIFAR-10 Image Classifier</h1>
<p>
A deep learning web app powered by <b>ResNet-18</b> trained on the CIFAR-10 dataset.
Upload an image or try real CIFAR-10 samples to see predictions in real-time.
</p>
</div>
""", unsafe_allow_html=True)


# -----------------------------
# Load model
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "resnet_cifar10.pth")

model = ResNet18().to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()
# -----------------------------
# Sidebar Navigation (STEP 2)
# -----------------------------
st.sidebar.title("📊 Navigation")
page = st.sidebar.radio(
    "Go to",
    ["Image Prediction", "Class-wise Accuracy"]
)
 
@st.cache_data
def compute_classwise_accuracy_cached(classes):
    

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Reload model INSIDE function (safe for caching)
    model = ResNet18().to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    # Load test dataset INSIDE function
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    testset = torchvision.datasets.CIFAR10(
        root="./data", train=False, download=True, transform=test_transform
    )

    testloader = torch.utils.data.DataLoader(
        testset, batch_size=128, shuffle=False
    )

    class_correct = [0] * len(classes)
    class_total = [0] * len(classes)

    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)

            for i in range(labels.size(0)):
                label = labels[i].item()
                pred = preds[i].item()
                if label == pred:
                    class_correct[label] += 1
                class_total[label] += 1

    return {
        classes[i]: round(100 * class_correct[i] / class_total[i], 2)
        for i in range(len(classes))
    }


# -----------------------------
# Classes
# -----------------------------
classes = ['airplane','automobile','bird','cat','deer',
           'dog','frog','horse','ship','truck']

# -----------------------------
# Image preprocessing
# -----------------------------
transform = transforms.Compose([
    transforms.Resize((32,32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
])

# -------------------------------
# -------------------------------
# -------------------------------
if page == "Image Prediction":

    st.markdown("## 🧪 Try with Sample CIFAR-10 Images")

    use_sample = st.checkbox("Use CIFAR-10 sample image")

    if use_sample:
        cifar_dataset = load_cifar_samples()

        col1, col2 = st.columns([1, 1])

        with col1:
            if st.button("🔄 Next Sample Image"):
                st.session_state.sample_idx = random.randint(
                    0, len(cifar_dataset) - 1
                )

        idx = st.session_state.sample_idx
        img_tensor, true_label = cifar_dataset[idx]

        # Convert tensor for display
        img_display = img_tensor.clone()
        img_display = img_display * 0.5 + 0.5
        img_display = img_display.permute(1, 2, 0).numpy()

        st.image(
            img_display,
            caption=f"Sample CIFAR-10 Image (True class: {classes[true_label]})",
            width=250
        )

        input_tensor = img_tensor.unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(input_tensor)
            probs = torch.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probs, 1)

        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("🔮 Model Prediction")

        conf = confidence.item() * 100

        st.markdown(
            f"<div class='pred'>Predicted Class: {classes[predicted.item()]}</div>",
            unsafe_allow_html=True
        )

        st.markdown(f"""
        <div class="conf-bar">
            <div class="conf-fill" style="width:{conf}%;"></div>
        </div>
        <p>{conf:.2f}% confidence</p>
        """, unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

    # -----------------------------
    # File uploader
    # -----------------------------
    st.markdown("## 📤 Or Upload Your Own Image")

    uploaded_file = st.file_uploader(
        "Upload an image", type=["jpg", "png", "jpeg"]
    )

    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_container_width=True)

        img_tensor = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(img_tensor)
            probs = torch.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probs, 1)

        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("🔮 Model Prediction")

        conf = confidence.item() * 100

        st.markdown(
            f"<div class='pred'>{classes[predicted.item()]}</div>",
            unsafe_allow_html=True
        )

        animated_confidence_bar(conf)


        st.markdown("</div>", unsafe_allow_html=True)


# =========================================================
# CLASS-WISE ACCURACY PAGE
# =========================================================
if page == "Class-wise Accuracy":

    st.title("📊 Class-wise Accuracy on CIFAR-10 Test Set")
    st.write(
        "This page shows how well the trained ResNet-18 model performs "
        "on each CIFAR-10 class."
    )

    with st.spinner("Computing class-wise accuracy..."):
        class_acc = compute_classwise_accuracy_cached(classes)

    acc_data = {
        "Class": list(class_acc.keys()),
        "Accuracy (%)": list(class_acc.values())
    }

    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("📈 Class-wise Accuracy Distribution")
    st.bar_chart(acc_data, x="Class", y="Accuracy (%)")
    st.markdown("</div>", unsafe_allow_html=True)

    with st.expander("📋 View Accuracy Table"):
        st.table(acc_data)


# =========================================================
# FOOTER
# =========================================================
st.markdown("""
<div class="footer">
Built with ❤️ using PyTorch & Streamlit<br>
ResNet-18 | CIFAR-10 | Deep Learning Portfolio Project
</div>
""", unsafe_allow_html=True)
