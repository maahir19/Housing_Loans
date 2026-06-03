# 🏦 Housing Loan Fraud Detection System

Anti-fraud detection dashboard built with Logistic Regression on 20 years of housing loan data.

## Live App
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-app-name.streamlit.app)

## Features
- Real-time fraud risk scoring for new loan applications
- Model performance dashboard (AUC, Recall, Precision)
- 5-fold cross-validation results
- Feature importance visualization

## Model Performance
| Metric | Score |
|--------|-------|
| AUC-ROC | 0.9419 |
| Recall | 86.6% |
| Precision | 82.1% |
| KFold Mean AUC | 0.9481 |

## How to run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Dataset
Place `homeloancsv.csv` in the same folder as `app.py`.
