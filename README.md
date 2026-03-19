#  Age Estimator using Transfer Learning (ResNet50)

This project predicts a person's age from facial images using a deep learning model based on **ResNet50 transfer learning**. It also converts predictions into **age ranges** for better interpretability and includes a **Streamlit web app** for real-time interaction.

---

##  Features

* Predicts age from face images (regression)
* Converts predictions into age ranges (Child, Teen, Adult, etc.)
* Transfer learning using ResNet50
* Fine-tuning for improved accuracy
* Interactive web app built with Streamlit

---

##  Model Overview

* Backbone: ResNet50 (pretrained on ImageNet)
* Approach: Transfer Learning + Fine-tuning
* Output: Single neuron (age regression)
* Dataset: UTKFace dataset

---

##  Results

* Mean Absolute Error (MAE): ~7.5 years
* Range Accuracy: ~59%

---

##  Project Structure

```
age-estimator/
│
├── app.py                  # Streamlit app
├── age_estimator.ipynb     # Training notebook
├── requirements.txt        # Dependencies
└── .gitignore
```

---

##  Installation

```bash
pip install -r requirements.txt
```

---

##  Run the App

```bash
streamlit run app.py
```

---

##  How It Works

1. Upload a face image
2. Model predicts age
3. Output shows:

   * Predicted age
   * Age range (Child, Teen, Adult, etc.)

---

##  Notes

* The dataset and trained model file are not included due to size limitations.
* To run the app, train the model first using the notebook.

---

## 🧠 Key Concepts Used

* Transfer Learning
* Convolutional Neural Networks (CNNs)
* ResNet50 Architecture
* Fine-tuning
* Regression vs Classification

---


