import streamlit as st
import pandas as pd
import pickle

st.set_page_config(page_title="Customer Churn Predictor", page_icon="📉", layout="centered")

with open("model.pkl", "rb") as f:
    saved = pickle.load(f)

model = saved["model"]
encoders = saved["encoders"]
target_le = saved["target_le"]
columns = saved["columns"]

st.title("📉 Customer Churn Predictor")
st.write(
    "Predicts the likelihood a telecom customer will churn, based on the "
    "**IBM Telco Customer Churn** dataset. Model: Random Forest "
    "(77% accuracy, 77% recall on churners)."
)
st.divider()

st.subheader("Enter customer details")

col1, col2 = st.columns(2)

with col1:
    gender = st.selectbox("Gender", ["Female", "Male"])
    senior = st.selectbox("Senior Citizen", ["No", "Yes"])
    partner = st.selectbox("Has Partner", ["No", "Yes"])
    dependents = st.selectbox("Has Dependents", ["No", "Yes"])
    tenure = st.slider("Tenure (months)", 0, 72, 12)
    phone_service = st.selectbox("Phone Service", ["No", "Yes"])
    multiple_lines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
    internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    online_security = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
    online_backup = st.selectbox("Online Backup", ["No", "Yes", "No internet service"])

with col2:
    device_protection = st.selectbox("Device Protection", ["No", "Yes", "No internet service"])
    tech_support = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])
    streaming_tv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
    streaming_movies = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])
    contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    paperless = st.selectbox("Paperless Billing", ["No", "Yes"])
    payment = st.selectbox("Payment Method", [
        "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
    ])
    monthly_charges = st.number_input("Monthly Charges ($)", 0.0, 200.0, 70.0)
    total_charges = st.number_input("Total Charges ($)", 0.0, 10000.0, 840.0)

if st.button("Predict Churn Risk", type="primary", use_container_width=True):
    raw = {
        "gender": gender, "SeniorCitizen": 1 if senior == "Yes" else 0,
        "Partner": partner, "Dependents": dependents, "tenure": tenure,
        "PhoneService": phone_service, "MultipleLines": multiple_lines,
        "InternetService": internet_service, "OnlineSecurity": online_security,
        "OnlineBackup": online_backup, "DeviceProtection": device_protection,
        "TechSupport": tech_support, "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies, "Contract": contract,
        "PaperlessBilling": paperless, "PaymentMethod": payment,
        "MonthlyCharges": monthly_charges, "TotalCharges": total_charges
    }

    row = {}
    for col in columns:
        if col in encoders:
            row[col] = encoders[col].transform([raw[col]])[0]
        else:
            row[col] = raw[col]

    X_input = pd.DataFrame([row])[columns]
    pred = model.predict(X_input)[0]
    proba = model.predict_proba(X_input)[0][1]

    st.divider()
    if pred == 1:
        st.error(f"⚠️ **High churn risk** — {proba*100:.1f}% probability of churn")
    else:
        st.success(f"✅ **Low churn risk** — {proba*100:.1f}% probability of churn")

    st.progress(float(proba))

    st.caption(
        "Top churn drivers in this model: Contract type, tenure, total charges, "
        "online security, and monthly charges."
    )

st.divider()
st.caption("Built by [Your Name] · [GitHub repo link] · Model trained on IBM Telco Customer Churn dataset")
