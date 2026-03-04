# SepsisGuard-AI
# SepsisGuard-AI: Predictive Clinical Monitoring Model
**Lead Developer:** Preethi R  
**Domain:** Biomedical AI / Clinical Decision Support Systems

## 🏥 Clinical Significance
Sepsis is a life-threatening condition caused by the body's extreme response to infection. Early detection is critical; every hour of delayed treatment increases mortality rates significantly. 

This project utilizes Machine Learning to predict sepsis probability by monitoring 7 key physiological parameters:
* **Heart Rate (HR)**
* **Oxygen Saturation (SpO2)**
* **Temperature**
* **Mean Arterial Pressure (MAP)**
* **Respiratory Rate**
* **Systolic & Diastolic Blood Pressure**

## 🛠️ Technical Workflow
### 1. Data Preprocessing & Balancing
Medical datasets are often imbalanced (fewer sepsis cases than healthy ones). To prevent model bias, I implemented **SMOTE (Synthetic Minority Over-sampling Technique)**. This ensures the model learns the specific signatures of septic onset rather than just predicting "healthy" for every patient.

### 2. Machine Learning Pipeline
* **Model:** XGBoost Classifier (Extreme Gradient Boosting)
* **Optimization:** Scikit-learn for feature scaling and validation.
* **Evaluation:** Focused on **Sensitivity (Recall)** to ensure no potential sepsis cases are missed by the system.



### 3. Deployment
The model is integrated with a **Gradio** web interface, allowing healthcare professionals to input vital signs and receive an instantaneous risk assessment and probability score.

## 🚀 How to Run
1. Open the `.ipynb` file in Google Colab using the badge at the top.
2. Install dependencies: `pip install -r requirements.txt`
3. Run the final cell to launch the Gradio interface.

## 📁 Repository Structure
* `Sepsis_Prediction_Model.ipynb`: Full research code and data visualization.
* `sepsis_model.pkl`: The trained XGBoost model.
* `requirements.txt`: List of necessary Python libraries.
