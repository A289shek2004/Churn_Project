# 📉 Telco Customer Churn Prediction

This project uses machine learning to predict customer churn in a telecom company. By analyzing customer behavior, tenure, contract types, and service usage, the model forecasts whether a customer is likely to cancel their subscription.

---

## 🚀 Project Highlights

- **Problem Statement:** Predict whether a customer will churn (leave the service) based on historical data.
- **Dataset:** [Telco Customer Churn Dataset](https://www.kaggle.com/blastchar/telco-customer-churn)
- **Model Used:** Logistic Regression
- **Evaluation Metrics:** Accuracy, Confusion Matrix, Classification Report

---

## 🗂️ Project Structure

## 🔍 Exploratory Data Analysis (EDA)

- Checked for missing values and data types
- Visualized relationships between:
  - Churn and Contract type
  - Tenure vs. Churn
  - Internet Service vs. Churn
  - Monthly Charges vs. Churn

---

## 🛠️ Data Preprocessing

- Converted categorical columns using Label Encoding and One-Hot Encoding
- Removed irrelevant or redundant features (e.g., `customerID`)
- Scaled numerical features where necessary

---

## 🧠 Model Training

- Trained a **Logistic Regression** model to classify customers into "Churn" or "No Churn"
- Saved the trained model as `churn_model.pkl` using `pickle`

---

## 📈 Evaluation

- Achieved solid accuracy on test data
- Evaluated with:
  - Confusion Matrix
  - Precision, Recall, F1-Score
  - ROC Curve (can be added as an improvement)

---

## 🔮 Future Improvements

- Test other models (Random Forest, XGBoost, SVM)
- Implement cross-validation and hyperparameter tuning
- Add a frontend using **Streamlit** or **Flask** for real-time churn prediction
- Deploy the model as an API using **FastAPI** or **Flask**

---

## 📦 Requirements

- `pandas`
- `numpy`
- `matplotlib`
- `seaborn`
- `scikit-learn`

Install all dependencies:
```bash
pip install -r requirements.txt

**✍️ Author**
Abhishek Gupta
Aspiring Data Scientist | Project Creator
