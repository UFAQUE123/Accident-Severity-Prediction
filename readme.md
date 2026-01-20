
# 🛣️ Accident Severity Prediction System

An end-to-end Machine Learning project that predicts **road accident severity** (Fatal Injury, Serious Injury, or Slight Injury) using accident dataset taken from kaggle. The link of the dataset is:
["/kaggle/input/road-traffic-severity-classification/RTA Dataset.csv"]

The project covers **data analysis, feature engineering, model training, evaluation**, and **deployment using Streamlit**.

---

## 📌 Project Overview

Road traffic accidents are a major public safety concern worldwide.
This project predicts accident severity based on driver, vehicle, road, and environmental conditions using Machine Learning techniques.

---

## 🎯 Objectives

- Perform Exploratory Data Analysis (EDA)
- Handle missing values and class imbalance
- Feature engineering and selection
- Train and tune ML models
- Deploy an interactive Streamlit app
- Display prediction confidence

---

## 🗂️ Project Structure

```
accident-severity-prediction/
│
├── dataset/
│   └── RTA_dataset.csv
│
├── notebooks/
│   └── Accident_Severity_Prediction.ipynb
│
├── trained_models/
│   ├── rta_model.joblib
│   ├── rta_tuned_rf.joblib
│   └── ordinal_encoder.joblib
│
├── app.py
├── requirements.txt
└── README.md
```

---

## 📊 Dataset

Contains road traffic accident records including:
- Driver demographics
- Vehicle information
- Road & environmental conditions
- Time & location
- Accident severity (target)

### Target Variable
- Fatal Injury
- Serious Injury
- Slight Injury

---

## 🔍 EDA Highlights

- More vehicles & casualties increase severity
- Night-time accidents are more severe
- Accident location & type matter most
- Road surface has limited impact
- Target classes are imbalanced

---

## ⚙️ Data Preprocessing

- Filled missing values with `Unknown`
- Extracted hour from time feature
- One-hot encoding (analysis)
- Ordinal encoding (deployment)
- Label encoding for target

---

## ⚖️ Imbalance Handling

- Used **SMOTENC** for categorical + numerical data

---

## 🧠 Feature Selection

- Mutual Information
- Chi-Square test
- Correlation analysis
- PCA (exploratory)

---

## 🤖 Model Training

- Baseline: Random Forest
- Tuned: Random Forest + Pipeline + GridSearchCV

### Best Performance
- **Weighted F1-score ≈ 80%**
- Cross-validated and stratified split

---

## 🚀 Streamlit App

Features:
- Model selection (Baseline / Tuned)
- Real-time predictions
- Confidence score display
- Clean UI with severity indicators

Run the app:
```bash
streamlit run app.py
```
## Conclusion

- The baseline Random Forest shows reasonable performance but is affected by class imbalance.
- The tuned Random Forest (700 trees, depth 20) achieved strong offline metrics but failed to generalize well in deployment.
- The SMOTENC + GridSearchCV pipeline delivers more stable and consistent predictions in real-world app usage by effectively handling categorical imbalance.

Hence, the SMOTENC-based pipeline is the most reliable and suitable model for deployment, despite a slightly lower weighted F1-score.

