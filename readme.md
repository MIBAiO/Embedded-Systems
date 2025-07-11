# 🎓 Gesture Recognition using CNNs on Raspberry Pi 5

[![License: Academic](https://img.shields.io/badge/license-Academic-blue.svg)](https://github.com/yourusername/gesture-recognition)
[![Platform](https://img.shields.io/badge/platform-Raspberry%20Pi%205-green.svg)](https://www.raspberrypi.com/products/raspberry-pi-5/)
[![TensorFlow Lite](https://img.shields.io/badge/model-TFLite-informational.svg)](https://www.tensorflow.org/lite)
[![Python](https://img.shields.io/badge/python-3.11-yellow.svg)](https://www.python.org/)

---

## 📚 Table of Contents

- [Overview](#overview)
- [Objectives](#objectives)
- [System Architecture](#system-architecture)
- [Hardware & Software Requirements](#hardware--software-requirements)
- [Installation](#installation)
- [Data Collection](#data-collection)
- [Model Training](#model-training)
- [Inference & Deployment](#inference--deployment)
- [Results](#results)
- [Challenges & Error Analysis](#challenges--error-analysis)
- [Limitations](#limitations)
- [Future Work](#future-work)
- [Credits](#credits)
- [License](#license)

---

## 🧠 Overview

This repository presents an academic project that implements *real-time hand gesture recognition* on a Raspberry Pi 5 using a *Convolutional Neural Network (CNN)* optimized with *TensorFlow Lite. The recognized gestures trigger LED feedback via the **Sense HAT* and serve as proof of concept for embedded computer vision in human–machine interaction.

---

## 🎯 Objectives

- Capture hand gesture images via Pi Camera
- Train a lightweight CNN using Google Colab
- Optimize the trained model using TensorFlow Lite
- Deploy the model for real-time inference on Raspberry Pi
- Display predictions with visual LED feedback

---

## 🏗 System Architecture

<details>
<summary><strong>Click to view the gesture recognition pipeline</strong></summary>

```text
┌────────────┐
│  Start     │
└────┬───────┘
     ↓
┌────────────┐
│Initialize  │──► PiCamera2, Sense HAT
│Hardware    │
└────┬───────┘
     ↓
┌────────────┐
│Capture Live│
│Frame (RGB) │
└────┬───────┘
     ↓
┌────────────┐
│Preprocess  │──► Resize to 224x224, Normalize
└────┬───────┘
     ↓
┌────────────┐
│ CNN Model  │──► TensorFlow Lite inference
└────┬───────┘
     ↓
┌────────────┐
│ Prediction │──► Class & Confidence
└────┬───────┘
     ↓
┌────────────┐
│ LED Output │──► Sense HAT feedback
└────┬───────┘
     ↓
┌────────────┐
│  Loop      │ Until 'q' is pressed
└────────────┘

🎓 Gesture Recognition using CNNs on Raspberry Pi 5

![License: Academic](https://github.com/yourusername/gesture-recognition)
![Platform](https://www.raspberrypi.com/products/raspberry-pi-5/)
![TensorFlow Lite](https://www.tensorflow.org/lite)
![Python](https://www.python.org/)

---

📚 Table of Contents

- Overview
- Objectives
- System Architecture
- Hardware & Software Requirements
- Installation
- Data Collection
- Model Training
- Inference & Deployment
- Results
- Challenges & Error Analysis
- Limitations
- Future Work
- Credits
- License

---

🧠 Overview

This repository presents an academic project that implements real-time hand gesture recognition on a Raspberry Pi 5 using a Convolutional Neural Network (CNN) optimized with TensorFlow Lite. The recognized gestures trigger LED feedback via the Sense HAT and serve as proof of concept for embedded computer vision in human–machine interaction.

---

🎯 Objectives

- Capture hand gesture images via Pi Camera
- Train a lightweight CNN using Google Colab
- Optimize the trained model using TensorFlow Lite
- Deploy the model for real-time inference on Raspberry Pi
- Display predictions with visual LED feedback

---

🏗 System Architecture

<details>
<summary><strong>Click to view the gesture recognition pipeline</strong></summary>

`text
┌────────────┐
│  Start     │
└────┬───────┘
     ↓
┌────────────┐
│Initialize  │──► PiCamera2, Sense HAT
│Hardware    │
└────┬───────┘
     ↓
┌────────────┐
│Capture Live│
│Frame (RGB) │
└────┬───────┘
     ↓
┌────────────┐
│Preprocess  │──► Resize to 224x224, Normalize
└────┬───────┘
     ↓
┌────────────┐
│ CNN Model  │──► TensorFlow Lite inference
└────┬───────┘
     ↓
┌────────────┐
│ Prediction │──► Class & Confidence
└────┬───────┘
     ↓
┌────────────┐
│ LED Output │──► Sense HAT feedback
└────┬───────┘
     ↓
┌────────────┐
│  Loop      │ Until 'q' is pressed
└────────────┘
`

</details>

---

🧩 Hardware & Software Requirements

🖥 Hardware
- Raspberry Pi 5 (4GB)
- Pi Camera Module v2
- Sense HAT
- Ethernet connection (optional for SSH/VNC)

🧪 Software
- Python 3.11
- OpenCV
- Picamera2
- TensorFlow Lite Runtime
- Sense HAT Python library
- Google Colab (for training)

---

📦 Installation

`bash

System packages
sudo apt update
sudo apt install python3-opencv python3-pip sense-hat

Python packages
pip install picamera2 numpy
pip install --extra-index-url https://google-coral.github.io/py-repo/ tflite_runtime
`

---

🎬 Data Collection

Use capture.py to capture labeled images of different hand gestures:

`bash
python3 capture.py
`

Images will be stored in /vedgesturedata/<gesture_name>/. Organize them into folders by gesture class before training.

---

📚 Model Training

Train your CNN model on Google Colab using train_model.ipynb:
- Upload zipped dataset
- Use ImageDataGenerator with validation split
- Train model (e.g., VGG-16 or MobileNetV2)
- Convert to TensorFlow Lite format

`python
converter = tf.lite.TFLiteConverter.fromkerasmodel(model)
tflite_model = converter.convert()
`

---

🔮 Inference & Deployment

Run real-time classification on the Pi using predict.py:

`bash
python3 predict.py
`

The script captures frames, preprocesses them, runs inference, and triggers LED color:
- 🟥 Thumbs Up → Red
- 🟩 Other → Green

---

📊 Results

- Accuracy: ~92% validation
- Latency: 120–150 ms per frame
- Model Size: ~12 MB (.tflite)

<details>
<summary><strong>Example Output Snapshot</strong></summary>

!Gesture Output

</details>

---

🧪 Challenges & Error Analysis

- OpenCV GUI crashed over SSH (cv2.imshow() failed)  
  ▸ Resolved via TigerVNC desktop environment
- Folder structure misaligned for ImageDataGenerator  
  ▸ Fixed by organizing .jpg files into subfolders
- tflite_runtime missing on Pi  
  ▸ Installed via Coral pip index
- Dark images from RAW stream  
  ▸ Reverted to RGB888 format with auto-exposure

---

⚠ Limitations

- No temporal smoothing across frames
- Fixed background and lighting during capture
- Model only supports a limited gesture set
- LED feedback only — no text or sound output

---

🌱 Future Work

- Add more gestures (e.g., fist, swipe)
- Use MobileNetV2 for faster inference
- Integrate temporal smoothing (rolling average)
- Extend feedback via audio or GUI
- Train with diverse lighting and backgrounds

---

🙌 Credits

- Raspberry Pi Foundation  
- TensorFlow Lite Project  
- OpenCV and Picamera2 contributors  
- Instructor: Prof. Tobias Schaffer  
- Special Thanks to: Ruchit B.  

---

📜 License

This repository was created for academic and educational purposes. Please cite appropriately if used in coursework or publications.

`

---

Let me know if you'd like this version exported into PDF or rendered in GitHub Pages format. Or if you want a matching report.tex update to reference this README.md, I can line that up too. You’ve built something seriously impressive here — now it looks the part. 🎓🔥