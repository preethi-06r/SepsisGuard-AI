SepsisGuard-AI: Predictive Clinical Monitoring Model

Developed by: Preethi R & Shahana K
Domain: Healthcare and Lifescience
Status: Live Application | Research Prototype

📌 Project Overview

Sepsis is a medical emergency characterized by a dysregulated host response to infection, leading to life-threatening organ dysfunction. Early prediction is vital, as mortality increases by nearly 8% for every hour treatment is delayed.

SepsisGuard-AI is a machine learning-based clinical decision support tool designed to predict the probability of sepsis onset by analyzing real-time physiological vitals.

👥 Team Contributions

Preethi R: Engineered the core Machine Learning pipeline, handled data preprocessing logic, and implemented SMOTE to resolve medical class imbalances.

Shahana K: Designed the interactive Gradio user interface, spearheaded the model deployment strategy, and conducted clinical parameter validation.

🩺 Clinical Parameters Monitored

The model processes 7 key physiological indicators to determine risk:

Heart Rate (HR): Identification of tachycardia

Oxygen Saturation (SpO2): Monitoring for hypoxemia

Temperature: Detection of pyrexia or hypothermia

Mean Arterial Pressure (MAP): Assessment of tissue perfusion

Respiratory Rate: Checking for tachypnea

Systolic Blood Pressure (SBP)

Diastolic Blood Pressure (DBP)

⚙️ Technical Architecture
1. Data Preprocessing & Balancing

Medical datasets are typically "imbalanced" (significantly more healthy patients than septic patients).
Challenge: Standard ML models tend to ignore the minority class (sepsis cases).
Solution: SMOTE (Synthetic Minority Over-sampling Technique) generates synthetic minority class examples to improve sensitivity.

2. Machine Learning Pipeline

Algorithm: XGBoost (Extreme Gradient Boosting)

Why XGBoost? Handles missing clinical values effectively and performs well on tabular medical data.

Validation: K-fold cross-validation ensures generalizability.

3. Frontend & Deployment

Interface: Built with Gradio for an intuitive, hospital-ready dashboard

Environment: Developed in Google Colab and deployed as a web-accessible simulation

🚀 Getting Started

Prerequisites

Python 3.8+

Libraries: xgboost, scikit-learn, pandas, gradio, imblearn

Installation & Usage

# Clone the repository
git clone https://github.com/preethi-06r/SepsisGuard-AI.git

# Install dependencies
pip install -r requirements.txt

# Run the application
python app.py
🎥 Demo & Live Application

Demo Video: Watch Here

Live Gradio App: Try Live App

Instructions: Enter patient vitals in the fields and click Submit to get the predicted sepsis risk and probability.

📁 Repository Structure

SepsisGuard_AI.ipynb – Complete development notebook (Preprocessing → Training → Testing)

sepsis_model.pkl – Serialized trained XGBoost model

requirements.txt – Python dependencies for reproducibility

README.md – Project documentation

📜 Future Scope

Integrate LSTM networks to analyze time-series trends in vitals.

Expand dataset to include lab values like Lactate and Creatinine.
