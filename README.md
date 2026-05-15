# 🥕🍎 Fruits & Vegetables Disease Detection System

An AI-powered Deep Learning web application that detects whether fruits and vegetables are **Healthy** or **Rotten** using Computer Vision and Convolutional Neural Networks (CNN).

---

## 🚀 Project Overview

This project uses a trained CNN model to classify fruits and vegetables from uploaded images.  
The system predicts the condition of the item and displays the result with a confidence score through a modern responsive web interface.

The project was developed to demonstrate the practical implementation of:

- Deep Learning
- Image Classification
- Computer Vision
- Flask Web Development

---

## ✨ Features

✅ Upload fruit or vegetable images  
✅ Detect Healthy or Rotten condition  
✅ Real-time prediction using CNN model  
✅ Confidence score display  
✅ Responsive modern UI  
✅ Glassmorphism-based frontend design  
✅ Multiple fruits & vegetables support  

---

## 🧠 Supported Classes

| Fruits & Vegetables | Classes |
|---|---|
| Apple | Healthy / Rotten |
| Banana | Healthy / Rotten |
| Bellpepper | Healthy / Rotten |
| Carrot | Healthy / Rotten |
| Cucumber | Healthy / Rotten |
| Grape | Healthy / Rotten |
| Guava | Healthy / Rotten |
| Jujube | Healthy / Rotten |
| Mango | Healthy / Rotten |
| Orange | Healthy / Rotten |
| Pomegranate | Healthy / Rotten |
| Potato | Healthy / Rotten |
| Strawberry | Healthy / Rotten |
| Tomato | Healthy / Rotten |

---

## 🛠️ Technologies Used

### Programming Languages
- Python
- HTML5
- CSS3
- JavaScript

### Libraries & Frameworks
- TensorFlow
- Keras
- Flask
- NumPy
- Matplotlib

### Tools
- VS Code
- Git & GitHub
- Kaggle Dataset
- Python Virtual Environment (venv)

---

## 🧩 Model Architecture

The project uses a **Convolutional Neural Network (CNN)** consisting of:

- Convolution Layers
- MaxPooling Layers
- Flatten Layer
- Dense Layers
- Dropout Layer
- Softmax Output Layer

---

## 📂 Project Structure

```bash
Fruits-Vegetables-Disease-Detection/
│
├── app.py
├── predict.py
├── train_model.py
├── requirements.txt
├── README.md
│
├── templates/
│   ├── index.html
│   └── result.html
│
├── static/
├── uploads/
├── model/
└── dataset/
```

---

## ⚙️ Installation & Setup

### 1️⃣ Clone Repository

```bash
git clone https://github.com/mrinal-daharia/Fruits-Vegetables-Disease-Detection.git
```

---

### 2️⃣ Open Project Folder

```bash
cd Fruits-Vegetables-Disease-Detection
```

---

### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

### 4️⃣ Train Model

```bash
python train_model.py
```

---

### 5️⃣ Run Flask Application

```bash
python app.py
```

---

### 6️⃣ Open in Browser

```bash
http://127.0.0.1:5000
```

---

## 📊 Model Performance

- Validation Accuracy: ~85%
- Multi-class Image Classification
- CNN-based Deep Learning Model
- Real-time Prediction System

---

## 🔍 How It Works

1. User uploads an image.
2. The image is resized and normalized.
3. CNN extracts visual features.
4. Model compares learned patterns.
5. Prediction is generated.
6. Result with confidence score is displayed.

---

## 🔍 Problems Solved

✔️ Automated rotten food detection  
✔️ Reduced manual inspection  
✔️ Faster quality analysis  
✔️ Food wastage reduction support  
✔️ Smart agriculture assistance  

---

## ⚠️ Challenges Faced

- Large dataset handling
- Background noise in images
- Class imbalance
- Low confidence predictions
- UI responsiveness issues
- Overfitting prevention

---

## 💡 Future Improvements

🚀 Real-time camera detection  
🚀 Mobile application version  
🚀 Cloud deployment  
🚀 EfficientNet / MobileNet integration  
🚀 Disease treatment recommendations  
🚀 Multilingual support  
🚀 Advanced AI accuracy improvements  

---

## 📁 Dataset

This project uses the **Fruit and Vegetable Disease (Healthy vs Rotten)** dataset from Kaggle.

Dataset Link:  
https://www.kaggle.com/datasets/muhammad0subhan/fruit-and-vegetable-disease-healthy-vs-rotten

### Dataset Details
- 28 Classes
- 14 Fruits & Vegetables
- Healthy & Rotten Categories
- Image Classification Dataset
- Used for Deep Learning & Computer Vision tasks

> Dataset size is approximately 4.89 GB, therefore it is not included in this repository.

---

## 📸 Screenshots

### Home Page
- Upload fruits & vegetables images
- Modern responsive UI

### Prediction Page
- Displays detected class
- Shows confidence score
- Preview image support

---

## 👨‍💻 Author

**Mrinal Daharia**

GitHub:  
https://github.com/mrinal-daharia

---

## 📜 License

This project is developed for educational and learning purposes.
