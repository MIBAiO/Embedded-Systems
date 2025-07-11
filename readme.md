# 🤖 Gesture Recognition using CNN on Raspberry Pi 5

This project implements a real-time hand gesture recognition system using a Raspberry Pi 5, Pi Camera Module, and a TensorFlow Lite CNN model. The system captures live video, classifies gestures (thumbs up and other sign), and provides visual feedback using the Sense HAT's LED matrix.

---
## 🛠 Project Highlights

- 🎥 Real-time image capture via PiCamera2
- 🧠 CNN inference with TensorFlow Lite
- 🎨 LED feedback using Sense HAT
- 📦 Lightweight and portable deployment
- 📡 Data collection + training pipeline
- 📊 Accuracy up to 92% with low latency

---

## 🛠️ Hardware & Software

### Hardware
- Raspberry Pi 5 (4GB or higher)
- Pi Camera Module v2
- Sense HAT
- Ethernet or Wi-Fi connection

### Software
- Python 3.11
- TensorFlow Lite Runtime
- OpenCV
- Picamera2
- Sense HAT Python library

---

## 📁 Project Structure
embedded-systems/ 
├── ved_gesture_data/         # Collected gesture images 
├── capture.py                # Script to collect gesture images
├── train_model.ipynb         # Google Colab notebook for training
├── cnn_gesture_model.tflite  # Trained TFLite model
├── predict.py                # Real-time prediction script 
├── README.md                 # Project documentation 
└── report.tex                

---

## 🚀 Getting Started

### 1. Clone the repository
```bash
    git clone https://github.com/yourusername/gesture-recognition-pi.git
    cd gesture-recognition-pi
    sudo apt update
    sudo apt install python3-opencv python3-pip sense-hat
    pip install --extra-index-url https://google-coral.github.io/py-repo/ tflite_runtime
    pip install numpy picamera2### 
'''
### 2. Install dependencies on Raspberry Pi
