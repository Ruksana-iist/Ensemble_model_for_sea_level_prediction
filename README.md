# **Sea Level Prediction Using Stacked Ensemble Machine Learning**

## **Overview**
This repository contains an advanced **ensemble machine learning model** for predicting sea level variations. The model leverages multiple techniques, including **Random Forest, Gradient Boosting (XGBoost, CatBoost), and ElasticNet**, combined using a **stacked ensemble approach**. Hyperparameter optimization is performed using **Optuna**, and **SHAP analysis** is employed for feature importance evaluation.  

🚀 **Key Highlights:**
- **Stacked ensemble model** combining multiple machine learning algorithms.
- **Time-series cross-validation** ensures robust model performance.
- **Hyperparameter tuning with Optuna** for optimal settings.
- **SHAP analysis** for model explainability.
- **Demo dataset used** (original dataset is confidential from INCOIS).

![Tide gauge locations](Tide_gauge_locations.png)

---

## **Table of Contents**
1. [Introduction](#introduction)
2. [Implementation Details](#implementation-details)
3. [Model Architecture](#model-architecture)
4. [Results](#results)
5. [Files](#files)
6. [Repository Structure](#repository-structure)
7. [Future Enhancements](#future-enhancements)

---

## **Introduction**
Sea level rise is a critical challenge in coastal management and climate research. Traditional models often suffer from **high variance** and **bias** due to complex ocean dynamics. **This project applies ensemble learning to enhance prediction accuracy and stability.**  

### **Why Ensemble Learning?**
✅ Reduces model variance by combining multiple learners.  
✅ Improves robustness against overfitting.  
✅ Captures complex relationships between oceanographic variables.  

### **Project Objectives:**
🔹 Predict sea level variations using a data-driven approach.  
🔹 Implement **bagging, boosting, and stacking** ensemble methods.  
🔹 Optimize hyperparameters for **high predictive accuracy**.  

---

## **Implementation Details**
The **stacked ensemble model** integrates predictions from **multiple base learners** and combines them using a **meta-learner (ElasticNet)**.  

### **Machine Learning Models Used:**
- **Base Models:**
  - Random Forest 🌳
  - XGBoost ⚡
  - CatBoost 🐱📊
  - ElasticNet 📈

- **Ensemble Techniques:**
  - **Bagging** (Bootstrap Aggregating)
  - **Boosting** (XGBoost, CatBoost)
  - **Stacking** (Meta-learning for optimal predictions)

### **Pipeline Workflow**
1. **Data Preprocessing:** Handling missing values, feature scaling (StandardScaler), and feature engineering (moving averages, lag variables).  
2. **Model Training:** Base models are trained and optimized with **Optuna**.  
3. **Ensemble Integration:** Predictions from base models are combined using **stacking**.  
4. **Evaluation:** Performance is measured using **MAE, RMSE, and R² score**.  
5. **Model Interpretability:** **SHAP analysis** provides insights into feature importance.  
6. **Prediction & Visualization:** Generates **sea level forecasts** and visualizes trends.  

---

## **Model Architecture**
```mermaid
graph TD;
    A[Input Data] -->|Feature Engineering| B[Train Base Models];
    B --> C[Random Forest];
    B --> D[XGBoost];
    B --> E[CatBoost];
    C --> F[Ensemble Stacking];
    D --> F;
    E --> F;
    F --> G[Meta Learner (ElasticNet)];
    G --> H[Final Prediction];
---

## **Results**
📊 **Performance Metrics:**  
- **Mean Absolute Error (MAE):** *TBD*  
- **Root Mean Squared Error (RMSE):** *TBD*  
- **R² Score:** *TBD*  

🔍 **Feature Importance (SHAP Analysis):**  
- `Feature_1` → **Most influential variable**  
- `Feature_2` → **Moderate impact**  
- `Feature_3` → **Least impact**  

📌 **Insights:**
- Stacking improves prediction accuracy over single models.
- SHAP analysis helps in identifying the most significant predictors.

---

## **Files**
📂 **Key Files:**
- `train_model.py` → Train and save the ensemble model.
- `model_utils.py` → Helper functions for training and evaluation.
- `config.yaml` → Stores hyperparameter configurations.
- `data/demo_sea_level.csv` → **Demo dataset** (Original INCOIS dataset is confidential).
- `README.md` → This documentation.

---

## **Repository Structure**
```bash
📦 sea-level-ensemble
 ┣ 📂 data
 ┃ ┗ 📜 demo_sea_level.csv  # Demo dataset
 ┣ 📂 models
 ┃ ┗ 📜 saved_model.pkl     # Trained model
 ┣ 📂 scripts
 ┃ ┣ 📜 train_model.py      # Training script
 ┃ ┗ 📜 model_utils.py      # Utility functions
 ┣ 📜 config.yaml           # Hyperparameters
 ┣ 📜 requirements.txt      # Dependencies
 ┗ 📜 README.md             # Documentation
```


📝 **Note:** This repository uses a **demo dataset** due to confidentiality agreements with **INCOIS**. The original dataset cannot be shared publicly. However, the methodology and model can be reproduced using similar data sources.

