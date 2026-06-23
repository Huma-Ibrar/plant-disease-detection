# 🌿 Plant Disease Recognition System with Multi-Language Support

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/Framework-Flask-lightgrey.svg)](https://flask.palletsprojects.com/)
[![Deep Learning](https://img.shields.io/badge/Deep%20Learning-TensorFlow%20%2F%20Keras-orange.svg)](https://www.tensorflow.org/)
[![Computer Vision](https://img.shields.io/badge/Computer%20Vision-OpenCV-red.svg)](https://opencv.org/)

An enterprise-grade Machine Learning and Deep Learning web application designed to classify plant leaf diseases from user-uploaded images using an optimized Convolutional Neural Network (CNN). 

Unlike academic prototypes, this system bridges the gap between research and production by implementing robust computer vision guardrails to validate image quality and providing a seamless, accessible user experience for localized farming communities.

---

## 📌 Table of Contents
* [Key Features](#-key-features)
* [Intelligent Image Validation (Guardrails)](#-intelligent-image-validation-guardrails)
* [Multi-Language Architecture](#-multi-language-architecture)
* [Technical Stack](#-technical-stack)
* [Project Structure](#-project-structure)
* [Installation & Environment Setup](#-installation--environment-setup)
* [Model Setup & Weights Deployment](#-model-setup--weights-deployment)
* [Running the Application](#-running-the-application)
* [Core Production Enhancements](#-core-production-enhancements)
* [Acknowledgment](#-acknowledgment)

---

## ✨ Key Features
* **High-Accuracy CNN Inference:** Rapid plant disease classification utilizing a fine-tuned deep learning model architecture.
* **Production-Ready Guardrails:** Integrated OpenCV validation to minimize false positives caused by sub-optimal inputs (e.g., automatically rejecting blurry uploads).
* **Bilingual Localization (Urdu & English):** Multi-language user interface support, making the diagnostic reports accessible to local farmers and global users alike.
* **Secure Session Handling:** Employs UUID-based filename masking to eliminate file system race conditions and collisions.
* **Responsive UI:** Clean, human-centric web frontend developed natively using Flask templates.

---

## 🛡️ Intelligent Image Validation (Guardrails)

To maximize real-world classification confidence and avoid processing corrupted/garbage data, the backend enforces strict image pre-validation protocols:

### 1. Advanced Blur Detection (Laplacian Variance)
The pipeline calculates the focus metric using the Laplacian operator's variance. If a user uploads an out-of-focus or shaky image, the system instantly catches it, halts inference, and alerts the user in their preferred language to upload a clearer image.

```python
import cv2

def calculate_blur_score(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()

# Production Threshold Enforcement
if blur_score < 80:
    return {
        "en": "Inference Rejected: Image focus score is too low. Please upload a clearer image.",
        "ur": "تصدیق مسترد: تصویر دھندلی ہے۔ براہ کرم واضح تصویر اپلوڈ کریں۔"
    }
2. Background Noise & Context Validation
Secondary structural filters ensure that the input frame contains adequate target features (leaf margins and surfaces) while discarding excessive background clutter or unrelated noise.

🌐 Multi-Language Architecture
To democratize AI utility in agriculture, the system features localized state handling:

English Interface: Tailored for researchers, developers, and global deployment contexts.

Urdu Interface: Tailored for local field workers and farmers to ensure actionability of the health diagnostic output without a language barrier.

🏗️ Technical Stack
Backend Framework: Flask (Python)

Computer Vision Processing: OpenCV-Python

Deep Learning Framework: TensorFlow / Keras

Numerical Operations: NumPy

Data Serialization: JSON

File Management: UUID Architecture

📁 Project Structure
The repository strictly decouples configuration, static distributions, view templates, and execution scripts:

Plaintext
plant-disease-detection/
│
├── models/
│   └── .gitkeep                         # Placeholder (Actual weights managed via external deployment)
│
├── static/
│   ├── css/                             # Component stylesheets
│   ├── images/                          # Native UI graphic assets
│   └── js/                              # Client-side form interceptors
│
├── templates/
│   └── home.html                        # Main interface viewport
│
├── .gitignore                           # Excludes local research logs, dataset docs & .keras arrays
├── README.md                            # Technical system specification
├── app.py                               # Core application controller & inference pipeline
├── plant_disease.json                   # Class mapping references
└── requirements.txt                     # Deterministic dependency manifest
⚙️ Installation & Environment Setup
Clone the Repository:

Bash
   git clone [https://github.com/Huma-Ibrar/plant-disease-detection.git](https://github.com/Huma-Ibrar/plant-disease-detection.git)
   cd plant-disease-detection
Configure Virtual Environment:

Bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On macOS/Linux:
   source venv/bin/activate
Install Dependencies:

Bash
   pip install -r requirements.txt
📥 Model Setup & Weights Deployment
Note: Due to standard remote asset host file size limits, the deployment-grade binary weights file (plant_disease_recog_model_pwp.keras) is provisioned externally.

Download Weights File: Obtain the serialized Keras artifact via our managed storage node:

👉 Download Pre-trained Model Weights (Ensure file is saved exactly as plant_disease_recog_model_pwp.keras)

Initialize Model Matrix Directory:

Bash
   mkdir models
Deploy Artifact: Transfer the downloaded .keras file directly into the newly provisioned models/ directory.

🚀 Running the Application
Verify Asset Configuration: Ensure that line execution parameters inside app.py resolve appropriately against the localized model array path:

Python
   tf.keras.models.load_model("models/plant_disease_recog_model_pwp.keras")
Boot Web Daemon:

Bash
   python app.py
Access Service Interface: Launch your web browser and navigate to the loopback target endpoint displayed in your execution shell terminal (typically http://127.0.0.1:5000/).

🚀 Core Production Enhancements
Collision-Free File Handling: Replaced original explicit client-side file names with cryptographic uuid4() configurations to ensure sandbox isolation during simultaneous web access.

Strict Separation of Concerns: Abstracted model definitions, web routing engines, and raw mathematical transformations into a robust state handler loop in app.py.

Reproducible Dependencies: Locked absolute versions across core image processing binaries to guarantee compile-time stability across different operating systems.

🙌 Acknowledgment
This production iteration builds upon foundations originally contributed by Vivek Kumar, extending infrastructure boundaries to address deployment reliability, strict error bounding, and computational guardrails.