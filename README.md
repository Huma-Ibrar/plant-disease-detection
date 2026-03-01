# 🌿 Plant Disease Recognition System

 Enhanced with Computer Vision Guardrails for Production Reliability  

A Machine Learning and Deep Learning-based web application that detects plant diseases from leaf images using a Convolutional Neural Network (CNN).

This project goes beyond a basic research prototype by adding intelligent validation mechanisms to ensure reliable, high-confidence predictions in real-world usage.

---

## 📌 Table of Contents

- [About The Project](#-about-the-project)
- [Key Features](#-key-features)
- [Intelligent Image Validation](#-intelligent-image-validation)
- [Production-Level Enhancements](#-production-level-enhancements)
- [Technical Stack](#-technical-stack)
- [System Workflow](#-system-workflow)
- [Model Setup](#-model-setup)
- [Installation](#-installation)
- [Running the Application](#-running-the-application)
- [Project Structure](#-project-structure)
- [License](#-license)

---

## 📖 About The Project

The **Plant Disease Recognition System** is a Machine Learning and Deep Learning application designed to classify plant leaf diseases using a trained CNN model.

While many ML projects stop at model training, this system focuses on **AI reliability and production readiness** by integrating computer vision guardrails to handle real-world image quality issues.

Due to GitHub file size limitations, the trained model is hosted separately on Google Drive.

---

## ✨ Key Features

-  Plant disease detection from leaf images
-  CNN-based image classification
-  Intelligent blur detection using OpenCV
-  Background noise validation
-  UUID-based secure file handling
-  Fast and user-friendly web interface
-  Deployment-ready project structure

---

## Intelligent Image Validation

To improve prediction reliability, the system includes computer vision guardrails:

###  Blur Detection (OpenCV)

The application calculates the **Laplacian variance** of the uploaded image:

- Images with a focus score below **80** are automatically rejected.
- Prevents inaccurate predictions caused by blurry inputs.
- Ensures high-confidence model inference.

###  Background Noise Detection

Additional validation ensures that:
- The model focuses strictly on plant leaf regions.
- Irrelevant or noisy backgrounds do not affect classification results.

Example blur detection logic:

```python
import cv2

def calculate_blur_score(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()

if blur_score < 80:
    return "Image is too blurry. Please upload a clearer image."
```

---

##  Production-Level Enhancements

This project was refactored to bridge the gap between research and real-world deployment:

-  UUID-based image naming to prevent file collisions
-  Secure and unique image processing
-  Standardized `requirements.txt` for reproducibility
-  Modular backend structure
-  Clean separation of model loading and inference logic

---

## 🏗️ Technical Stack

- Python  
- Flask  
- OpenCV  
- TensorFlow / Keras  
- NumPy  
- UUID  
- Machine Learning & Deep Learning (CNN)

---

## 📊 System Workflow

1. User uploads plant leaf image  
2. Blur detection checks image clarity  
3. Background validation runs  
4. Image is assigned a unique UUID filename  
5. CNN model performs classification  
6. Disease prediction result is displayed  

---

## 📥 Model Setup

Since the trained model file is large, it is hosted on Google Drive.

### 🔗 Download Model

Download the pre-trained model here:  
https://drive.google.com/file/d/1MbLe0qYmWtAn9TQNLGjPNLwbCcj5cpEk/view?usp=drive_link

⚠ Ensure the file access is set to **"Anyone with the link → Viewer"** before downloading.

---

### 📂 Step 1: Create Models Folder

Navigate to the project root directory and create a folder named `models` if it does not exist:

```bash
mkdir models
```

---

### 📦 Step 2: Move the Model File

Move the downloaded model file into the `models` directory:

```bash
mv /path/to/downloaded/model models/
```

Replace `/path/to/downloaded/model` with your actual file location.

---

### ✅ Step 3: Verify Model Placement

```bash
ls models
```

You should see your model file listed.

---

## ⚙ Installation

Clone the repository:

```bash
git clone https://github.com/Huma-Ibrar/plant-disease-detection.git
cd plant-disease-detection
```

Install required dependencies:

```bash
pip install -r requirements.txt
```

---

## 🚀 Running the Application

### 1️⃣ Update Model Path

Open `app.py` and locate:

```python
tf.keras.models.load_model("")
```

Update it to:

```python
tf.keras.models.load_model("models/your_model_file.keras")
```

Replace `your_model_file.keras` with your actual model filename.

---

### 2️⃣ Start the Server

```bash
python app.py
```

---

### 3️⃣ Access the Application

Open the URL displayed in the terminal in your web browser.

---

## 📁 Project Structure

```
plant-disease-detection/
│
├── app.py
├── models/
├── static/
├── templates/
├── requirements.txt
└── README.md
```

---

## 📜 License

This project is developed for educational and research purposes.

---

## 🙌 Acknowledgment

This project builds upon the open-source work of Vivek Kumar and has been enhanced to improve reliability, robustness, and production readiness.
