
import streamlit as st
import pandas as pd
import joblib


st.set_page_config(
    page_title="Intelligent Fraud Detection",
    page_icon="🔍",
    layout="wide"
)


@st.cache_resource
def load_model():
    return joblib.load("fraud_detection_model.pkl")

@st.cache_data
def load_data():
    return pd.read_csv("creditcard.csv")

model = load_model()
data = load_data()

st.title("🔍 Intelligent Fraud Detection System")
st.write("Credit Card Fraud Detection using Machine Learning")
st.divider()


st.subheader("Transaction Overview")

col1, col2, col3 = st.columns(3)

col1.metric("Total Transactions", f"{len(data):,}")
col2.metric("Fraudulent Transactions", f"{int(data['Class'].sum()):,}")
col3.metric("Genuine Transactions", f"{int((data['Class'] == 0).sum()):,}")

st.divider()

st.subheader("Model Performance")

col1, col2, col3, col4 = st.columns(4)

col1.metric("Accuracy", "99.96%")
col2.metric("Precision", "94.12%")
col3.metric("Recall", "81.63%")
col4.metric("F1 Score", "87.43%")

st.divider()

st.subheader("Check a Transaction")

st.write("Select a transaction from the dataset to check the model prediction.")

transaction_id = st.number_input(
    "Enter Transaction Row Number",
    min_value=0,
    max_value=len(data) - 1,
    value=0,
    step=1
)

if st.button("Predict Transaction"):

   
    transaction = data.drop(columns=["Class"]).iloc[[int(transaction_id)]]

   
    prediction = model.predict(transaction)[0]
    probability = model.predict_proba(transaction)[0][1]

    st.subheader("Prediction Result")

    if prediction == 1:
        st.error("⚠️ Potential Fraudulent Transaction")
    else:
        st.success("✅ Genuine Transaction")

    st.write(f"Fraud Probability: {probability * 100:.2f}%")

    st.write("Actual Transaction Class:", 
             "Fraud" if data.iloc[int(transaction_id)]["Class"] == 1 else "Genuine")

st.divider()

st.caption("Intelligent Fraud Detection System | Machine Learning Project")
