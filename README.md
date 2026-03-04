# SepsisGuard-AI: Predictive Clinical Monitoring Model

**Developed by:** Preethi R & Shahana K  
**Domain:** Biomedical AI / Clinical Decision Support Systems  
**Status:** Live Application | Research Prototype

---

## 📌 Project Overview
Sepsis is a medical emergency characterized by a dysregulated host response to infection, leading to life-threatening organ dysfunction. Early prediction is vital, as mortality increases by nearly 8% for every hour treatment is delayed.

**SepsisGuard-AI** is a machine learning-based clinical decision support tool designed to predict the probability of sepsis onset by analyzing real-time physiological vitals.

## 👥 Team Contributions
* **Preethi R:** Engineered the core Machine Learning pipeline, handled data preprocessing logic, and implemented **SMOTE** to resolve medical class imbalances.
* **Shahana K:** Designed the interactive **Gradio** user interface, spearheaded the model deployment strategy, and conducted clinical parameter validation.

---

## 🩺 Clinical Parameters Monitored
The model processes 7 key physiological indicators to determine risk:
1.  **Heart Rate (HR):** Identification of tachycardia.
2.  **Oxygen Saturation (SpO2):** Monitoring for hypoxemia.
3.  **Temperature:** Detection of pyrexia or hypothermia.
4.  **Mean Arterial Pressure (MAP):** Assessment of tissue perfusion.
5.  **Respiratory Rate:** Checking for tachypnea.
6.  **Systolic Blood Pressure (SBP)**
7.  **Diastolic Blood Pressure (DBP)**



---

## ⚙️ Technical Architecture

### 1. Data Preprocessing & Balancing
Medical datasets are typically "imbalanced" (significantly more healthy patients than septic patients). 
* **Challenge:** Standard ML models tend to ignore the minority class (sepsis cases).
* **Solution:** We utilized **SMOTE (Synthetic Minority Over-sampling Technique)** to generate synthetic examples of the minority class, ensuring the model is highly sensitive to actual sepsis onset.

### 2. Machine Learning Pipeline
* **Algorithm:** XGBoost (Extreme Gradient Boosting).
* **Why XGBoost?** It handles missing clinical values effectively and provides superior performance on tabular medical data compared to standard neural networks.
* **Validation:** Implemented K-fold cross-validation to ensure model generalizability.

### 3. Frontend & Deployment
* **Interface:** Built with **Gradio** for an intuitive, hospital-ready dashboard.
* **Environment:** Developed in Google Colab and deployed as a web-accessible simulation.



---

## 🚀 Getting Started

### Prerequisites
* Python 3.8+
* Libraries: `xgboost`, `scikit-learn`, `pandas`, `gradio`, `imblearn`

### Installation & Usage
1.  **Clone the Repository:**
    ```bash
    git clone [https://github.com/preethi-06r/SepsisGuard-AI.git](https://github.com/preethi-06r/SepsisGuard-AI.git)
    ```
2.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
3.  **Run the Application:**
    ```bash
    python app.py
    ```

---

## 📁 Repository Structure
* `SepsisGuard_AI.ipynb`: The complete development notebook (Preprocessing -> Training -> Testing).
* `sepsis_model.pkl`: The serialized trained XGBoost model.
* `requirements.txt`: List of Python dependencies for reproducibility.
* `README.md`: Project documentation.

## 📜 Future Scope
* Integration of **LSTM (Long Short-Term Memory)** networks to analyze time-series trends in vitals.
* Expansion of the dataset to include laboratory values like Lactate and Creatinine.
