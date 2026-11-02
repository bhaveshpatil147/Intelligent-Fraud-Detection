# Intelligent Fraud Detection Dashboard

An interactive Credit Card Fraud Detection Dashboard built using Machine Learning and Streamlit.

## Features
- Interactive fraud prediction
- Fraud probability display
- Transaction statistics
- Model performance metrics
- Random Forest classification model

## Technologies Used
- Python
- Streamlit
- Pandas
- Scikit-learn
- Joblib

## Model Performance

| Metric | Score |
|---|---:|
| Accuracy | 99.96% |
| Precision | 94.12% |
| Recall | 81.63% |
| F1 Score | 87.43% |
| ROC AUC | 96.30% |

## Dataset
Credit Card Fraud Detection dataset from Kaggle:

https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

Download `creditcard.csv` and place it in the same folder as `app.py`.

## Run Locally

Install dependencies:

```bash
pip install streamlit pandas numpy scikit-learn joblib
```

Run the dashboard:

```bash
streamlit run app.py
```
