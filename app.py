import streamlit as st
import pandas as pd
import pickle
import plotly.graph_objects as go

st.set_page_config(page_title="Customer Churn Predictor", page_icon="📉", layout="wide")

# ---------------- Custom CSS ----------------
st.markdown("""
<style>
    .hero {
        padding: 1.8rem 2rem;
        border-radius: 16px;
        background: linear-gradient(135deg, #6E3AF2 0%, #B23AF2 100%);
        color: white;
        margin-bottom: 1.5rem;
    }
    .hero h1 { margin: 0; font-size: 2rem; }
    .hero p { margin: 0.4rem 0 0 0; opacity: 0.92; font-size: 0.95rem; }

    .metric-card {
        background: #1c1f2e;
        border: 1px solid #2d3142;
        border-radius: 12px;
        padding: 1rem 1.2rem;
        text-align: center;
    }
    .metric-card h3 { margin: 0; font-size: 1.6rem; color: #B23AF2; }
    .metric-card p { margin: 0.2rem 0 0 0; font-size: 0.8rem; color: #9aa0b0; }

    .result-box {
        border-radius: 14px;
        padding: 1.5rem;
        text-align: center;
        margin-top: 1rem;
    }
    .high-risk { background: rgba(255, 75, 75, 0.12); border: 1px solid #ff4b4b; }
    .low-risk  { background: rgba(60, 200, 130, 0.12); border: 1px solid #3cc882; }

    .stButton>button {
        border-radius: 10px;
        height: 3em;
        font-weight: 600;
        background: linear-gradient(135deg, #6E3AF2 0%, #B23AF2 100%);
        color: white;
        border: none;
    }
    .stButton>button:hover { opacity: 0.9; }

    .footer-note {
        text-align: center;
        color: #7a8091;
        font-size: 0.82rem;
        margin-top: 2rem;
    }
</style>
""", unsafe_allow_html=True)

# ---------------- Load model ----------------
with open("model.pkl", "rb") as f:
    saved = pickle.load(f)

model = saved["model"]
encoders = saved["encoders"]
target_le = saved["target_le"]
columns = saved["columns"]

# ---------------- Hero header ----------------
st.markdown("""
<div class="hero">
    <h1>📉 Customer Churn Predictor</h1>
    <p>Random Forest model trained on the IBM Telco Customer Churn dataset — predicts the probability a customer leaves, and why.</p>
</div>
""", unsafe_allow_html=True)

# ---------------- Model stat cards ----------------
c1, c2, c3, c4 = st.columns(4)
stats = [("77%", "Accuracy"), ("77%", "Recall (churners)"), ("54%", "Precision"), ("7,043", "Customers trained on")]
for col, (val, label) in zip([c1, c2, c3, c4], stats):
    with col:
        st.markdown(f"""<div class="metric-card"><h3>{val}</h3><p>{label}</p></div>""", unsafe_allow_html=True)

st.write("")

# ---------------- Input form ----------------
tab1, tab2 = st.tabs(["🧾 Customer Profile", "ℹ️ About this model"])

with tab1:
    left, right = st.columns(2)

    with left:
        st.markdown("**Demographics & Account**")
        gender = st.selectbox("Gender", ["Female", "Male"])
        senior = st.selectbox("Senior Citizen", ["No", "Yes"])
        partner = st.selectbox("Has Partner", ["No", "Yes"])
        dependents = st.selectbox("Has Dependents", ["No", "Yes"])
        tenure = st.slider("Tenure (months)", 0, 72, 12)
        contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
        paperless = st.selectbox("Paperless Billing", ["No", "Yes"])
        payment = st.selectbox("Payment Method", [
            "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
        ])

    with right:
        st.markdown("**Services**")
        phone_service = st.selectbox("Phone Service", ["No", "Yes"])
        multiple_lines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
        internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
        online_security = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
        online_backup = st.selectbox("Online Backup", ["No", "Yes", "No internet service"])
        device_protection = st.selectbox("Device Protection", ["No", "Yes", "No internet service"])
        tech_support = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])
        streaming_tv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
        streaming_movies = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])

    st.markdown("**Billing**")
    b1, b2 = st.columns(2)
    with b1:
        monthly_charges = st.number_input("Monthly Charges ($)", 0.0, 200.0, 70.0)
    with b2:
        total_charges = st.number_input("Total Charges ($)", 0.0, 10000.0, 840.0)

    st.write("")
    predict_clicked = st.button("🔮  Predict Churn Risk", type="primary", use_container_width=True)

    if predict_clicked:
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

        gcol, rcol = st.columns([1, 1.3])

        with gcol:
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=proba * 100,
                number={"suffix": "%", "font": {"size": 40}},
                gauge={
                    "axis": {"range": [0, 100], "tickcolor": "gray"},
                    "bar": {"color": "#B23AF2" if proba < 0.5 else "#ff4b4b"},
                    "steps": [
                        {"range": [0, 40], "color": "rgba(60,200,130,0.25)"},
                        {"range": [40, 70], "color": "rgba(255,193,7,0.25)"},
                        {"range": [70, 100], "color": "rgba(255,75,75,0.25)"},
                    ],
                },
                title={"text": "Churn Probability"}
            ))
            fig.update_layout(height=280, margin=dict(l=20, r=20, t=50, b=10),
                               paper_bgcolor="rgba(0,0,0,0)", font={"color": "white"})
            st.plotly_chart(fig, use_container_width=True)

        with rcol:
            risk_class = "high-risk" if pred == 1 else "low-risk"
            risk_label = "⚠️ High Churn Risk" if pred == 1 else "✅ Low Churn Risk"
            st.markdown(f"""
            <div class="result-box {risk_class}">
                <h2 style="margin-bottom:0.3rem;">{risk_label}</h2>
                <p style="opacity:0.85;">{proba*100:.1f}% predicted probability this customer churns</p>
            </div>
            """, unsafe_allow_html=True)

            st.write("")
            st.markdown("**Top drivers in this model:**")
            st.markdown("""
            - 📄 Contract type (month-to-month = highest risk)
            - ⏳ Tenure (newer customers churn more)
            - 💳 Total & monthly charges
            - 🔒 Online security add-on
            """)

with tab2:
    st.markdown("""
    ### About
    This app uses a **Random Forest Classifier** trained on the IBM Telco Customer Churn dataset
    (~7,000 customers). It's tuned to prioritize **recall** — catching as many true churners as
    possible — since in a real retention campaign, missing an at-risk customer costs more than
    a false alarm.

    **Business use case:** feed real customer records through this model monthly to flag
    high-risk accounts for a retention team to reach out to proactively.

    **Tools used:** Python, pandas, scikit-learn, Streamlit, Plotly
    """)

st.markdown('<p class="footer-note">Built by [Your Name] · '
            '<a href="#" style="color:#B23AF2;">GitHub repo</a> · '
            'Model trained on IBM Telco Customer Churn dataset</p>', unsafe_allow_html=True)
