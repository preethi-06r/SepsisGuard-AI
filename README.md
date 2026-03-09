# SepsisGuard-AI: Predictive Clinical Monitoring Model

**Developed by:** Preethi R & Shahana K  
**Domain:** Biomedical AI / Clinical Decision Support Systems  
**Status:** Live Application | Research Prototype  

---

## 📌 Project Overview
Sepsis is a medical emergency characterized by a dysregulated host response to infection, leading to life-threatening organ dysfunction. Early prediction is vital, as mortality increases by nearly **8% for every hour treatment is delayed**.

**SepsisGuard-AI** is a machine learning-based clinical decision support tool designed to predict the probability of sepsis onset by analyzing real-time physiological vitals.

---

## 🌐 Live Application
The SepsisGuard-AI model is deployed as an interactive web application where users can input patient vitals and receive a predicted sepsis risk assessment.

🔗 **Live Demo:**  
https://huggingface.co/spaces/Shahana1925/SepsisGuard.AI

---

## 👥 Team Contributions
**Preethi R**
- Engineered the core Machine Learning model
- Implemented data preprocessing
- Applied **SMOTE** to address medical class imbalance

**Shahana K**
- Designed the interactive **Gradio** user interface
- Led the model deployment process
- Conducted validation of clinical parameters

---

## 🩺 Clinical Parameters Monitored
The model processes **7 key physiological indicators** to determine sepsis risk:

1. **Heart Rate (HR)** – Identification of tachycardia  
2. **Oxygen Saturation (SpO₂)** – Monitoring for hypoxemia  
3. **Temperature** – Detection of fever or hypothermia  
4. **Mean Arterial Pressure (MAP)** – Assessment of tissue perfusion  
5. **Respiratory Rate** – Checking for tachypnea  
6. **Systolic Blood Pressure (SBP)**  
7. **Diastolic Blood Pressure (DBP)**  

---

## ⚙️ Technical Architecture

### 1. Data Preprocessing & Balancing
Medical datasets are typically **imbalanced**, meaning there are significantly more healthy patients than sepsis cases.

**Challenge:**  
Machine learning models may ignore the minority class.

**Solution:**  
We implemented **SMOTE (Synthetic Minority Over-sampling Technique)** to generate synthetic minority samples, improving model sensitivity to sepsis cases.

---

### 2. Machine Learning Model
**Algorithm:** XGBoost (Extreme Gradient Boosting)

**Why XGBoost?**
- Handles missing values effectively  
- Performs well on structured medical datasets  
- Provides strong predictive performance

**Validation:**  
K-Fold Cross Validation was used to ensure model generalization.

---

### 3. Frontend & Deployment
**Interface:** Gradio interactive dashboard  
**Development Environment:** Google Colab  
**Deployment Platform:** Hugging Face Spaces  

The application allows clinicians or researchers to input patient vitals and receive a predicted **sepsis risk probability**.

---

## 🚀 Getting Started

### Prerequisites
- Python 3.8+
- Required Libraries:
  - xgboost
  - scikit-learn
  - pandas
  - gradio
  - imblearn

---

### Installation & Usage

#### Clone the Repository
```bash
git clone https://github.com/preethi-06r/SepsisGuard-AI.git
```

#### Install Dependencies
```bash
pip install -r requirements.txt
```

#### Run the Application
```bash
python app.py
```

---

## 📁 Repository Structure

```
SepsisGuard-AI
│
├── SepsisGuard_AI.ipynb
├── sepsis_model.pkl
├── app.py
├── requirements.txt
└── README.md
```

---

## 📊 Potential Applications
- Early sepsis screening in hospitals  
- Clinical decision support systems  
- Remote patient monitoring  
- AI-assisted triage systems in emergency care  

---

## 🔬 Future Scope
- Integration of **LSTM (Long Short-Term Memory)** networks for time-series analysis of patient vitals  
- Inclusion of laboratory parameters such as **Lactate, Creatinine, and WBC count**  
- Validation using larger clinical datasets  
- Integration with hospital **Electronic Health Record (EHR)** systems  

---

## 📜 License
This project is developed as a **research prototype for educational and innovation purposes**.
