# 📉 Customer Churn Prediction

Predicts whether a telecom customer is likely to churn, using the IBM Telco Customer Churn dataset (~7,000 customers).

**🔗 Live demo:** [add your Streamlit URL here after deploying]

## Problem
Telecom companies lose significant revenue to customer churn. This project identifies at-risk customers *before* they leave, so retention teams can act early.

## Approach
- **Data cleaning:** handled missing `TotalCharges`, encoded categorical features
- **Model:** Random Forest Classifier (class-balanced, tuned for recall since missing a churner is costlier than a false alarm)
- **Result:** 77% accuracy, 77% recall on churners, 54% precision

## Key findings
Top 5 drivers of churn (by feature importance):
1. **Contract type** — month-to-month customers churn far more than 1-2 year contracts
2. **Tenure** — new customers (<12 months) are highest risk
3. **Total charges**
4. **Online security** — customers without it churn more
5. **Monthly charges**

## Business recommendation
Target retention offers at month-to-month customers in their first year, especially those without online security add-ons — this segment shows the highest churn probability.

## Tools
Python (pandas, scikit-learn), Streamlit (deployment), SQL (exploratory queries — see `/sql`)

## How to run locally
```bash
pip install -r requirements.txt
python train_model.py   # retrains the model
streamlit run app.py    # launches the web app
```

## Files
- `train_model.py` — data cleaning + model training
- `app.py` — Streamlit web app
- `model.pkl` — trained model + encoders
- `telco_churn.csv` — dataset
