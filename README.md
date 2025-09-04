# Dyslexia Prediction Web App

This project is a **machine learning–powered web application** for predicting dyslexia in students.  
It combines **Flask** (for the main website with subscription & contact forms) and **Streamlit** (for the interactive ML prediction tool).

## 🚀 Features
- **Dyslexia Prediction Tool** using an SVM model.
- **Streamlit interface** for easy input of features.
- **Flask website** with:
  - Landing page
  - Contact form (stores messages in MongoDB)
  - Subscription form (stores emails in MongoDB)
- Pre-trained model (`svm_model.pkl`) and encoders (`label_encoders.pkl`) included.
- Dataset included for retraining (`Dataset-Dyslexia-BCA (1).csv`).

---

## 📂 Project Structure

dys/
- │── app.py                 # Flask web app
- │── streamlit_app.py       # Streamlit ML tool
- │── dyslexia_model.py      # Model training script
- │── svm_model.pkl          # Saved trained model
- │── label_encoders.pkl     # Encoders for categorical features
- │── Dataset-Dyslexia-BCA.csv   # Training dataset
- │── templates/             # HTML templates (Flask)
- │── static/                # Static assets (CSS, JS, images)
- │── .venv/                 # Virtual environment (can be recreated)



---

## 🛠 Installation & Setup

### 1. Clone the repository
```bash
git clone https://github.com/yourusername/dyslexia-prediction.git
cd dyslexia-prediction/dys

📦 Requirements

Key libraries:

Flask
Streamlit
scikit-learn
pandas
imbalanced-learn
pymongo
