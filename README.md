# 🧠✨ NeuroCanvas

**NeuroCanvas** is a full-stack interactive neural network project that lets users **draw digits on a canvas**, train a simple neural network in real-time, and predict handwritten digits — all within the browser using a Python backend.

It's an end-to-end demo of how neural networks work behind the scenes — from pixel input to prediction — without using any machine learning libraries like TensorFlow or PyTorch.

---

## 🎯 Features

- ✍️ **Canvas-based Drawing**: Draw digits directly in the browser on a 20x20 grid
- 🧠 **Custom Neural Network**: Pure Python feedforward neural net implementation (no ML libraries!)
- 🔁 **Train & Predict**: Train the model in real-time via frontend and instantly test predictions
- 🛠 **Full-Stack Setup**:
  - **Frontend**: HTML5 + JavaScript Canvas
  - **Backend**: Python `http.server` with custom NN logic
- 💾 **Model Persistence**: Save and load weights to/from JSON
- 🧪 **Based on MNIST** (Optional): Easily switch to training on MNIST data

---

## 🚀 Demo

- checkout the ss or run it. 

---

## 📁 Folder Structure

1. ocr.py - has the offline training from the MNIST set
2. neural_network_design.py - neural network training (also test and predict)
3. server.py - (Backend) receives response after sending to the nn
4. ocr.html
5. ocr.js
6. Other loaded and saved files of the model and the backpropagated results fro optimised use

