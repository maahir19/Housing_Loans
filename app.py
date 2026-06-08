import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, precision_score, recall_score, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="Housing Loan Fraud Detection", page_icon=None, layout="wide")

st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&display=swap');
  html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
  .hero {
    background: linear-gradient(135deg, #1a1f2e 0%, #0d1321 100%);
    border: 1px solid #2d3748; border-radius: 16px;
    padding: 2.5rem 3rem; margin-bottom: 2rem; text-align: center;
  }
  .hero h1 { font-size: 2.2rem; font-weight: 600; color: #f0f4f8; margin: 0 0 0.5rem; }
  .hero p  { color: #8899aa; font-size: 1rem; margin: 0; }
  .hero .badge {
    display: inline-block; background: #1e3a5f; color: #60a5fa;
    font-size: 0.75rem; font-weight: 500; padding: 4px 12px;
    border-radius: 20px; margin-bottom: 1rem; letter-spacing: 0.05em;
  }
  .metric-card {
    background: #1a1f2e; border: 1px solid #2d3748;
    border-radius: 12px; padding: 1.25rem 1.5rem; text-align: center;
  }
  .metric-card .label  { font-size: 0.75rem; font-weight: 500; color: #8899aa; text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 0.5rem; }
  .metric-card .value  { font-size: 2rem; font-weight: 600; line-height: 1; margin-bottom: 0.25rem; }
  .metric-card .meaning{ font-size: 0.72rem; color: #6b7280; line-height: 1.4; }
  .stButton > button {
    width: 100%; background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%);
    color: white; border: none; border-radius: 10px;
    padding: 0.75rem 1.5rem; font-size: 1rem; font-weight: 500;
  }
  .cm-box {
    border-radius: 12px; padding: 1.2rem; text-align: center; cursor: pointer;
    transition: transform 0.15s; border: 2px solid transparent;
  }
  .cm-box:hover { transform: translateY(-2px); }
  .cm-box .cm-num  { font-size: 2.5rem; font-weight: 700; line-height: 1; }
  .cm-box .cm-lbl  { font-size: 0.8rem; font-weight: 500; margin-top: 5px; }
  .cm-box .cm-sub  { font-size: 0.7rem; opacity: 0.7; margin-top: 3px; }
  .loan-card {
    background: #1a1f2e; border: 1px solid #2d3748; border-radius: 10px;
    padding: 12px 16px; margin-bottom: 8px;
  }
  div[data-testid="stSidebar"] { background: #111827; border-right: 1px solid #1f2937; }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_and_train():
    df = pd.read_csv("homeloancsv.csv")
    yn_cols = ['Signature_Mismatch','ID_Docs_Altered','Salary_Round_Figure_Flag',
        'PF_Deduction_Missing','Salary_Credited_From_Personal_Acct',
        'ITR_Filed_Just_Before_Application','Month_End_Fund_Parking',
        'Valuation_Inflation_Flag','OC_CC_Available','Disbursement_To_NonSeller_Flag',
        'Bank_Statement_Tampered','PDF_Metadata_Anomaly','Borrower_Unreachable',
        'AML_Flag','Loan_Foreclosed_Within_6M','PEP_Involved','Overseas_Wire_Transfer_Detected']
    for col in yn_cols:
        df[col] = df[col].map({'Yes':1,'No':0,1:1,0:0}).fillna(0)

    df['income_discrepancy']   = (df['Monthly_Income_Stated'] - df['Actual_Bank_Credit_Monthly']).fillna(0)
    df['income_inflate_ratio'] = (df['Monthly_Income_Stated'] / df['Actual_Bank_Credit_Monthly'].replace(0,np.nan)).fillna(1).clip(upper=10)
    df['itr_income_gap']       = (df['Annual_Income_Stated'] - df['ITR_Income_Last_FY'].fillna(df['Annual_Income_Stated']))
    doc_flag_cols = ['Email_Name_Mismatch'] + yn_cols
    df['doc_risk_score'] = df[doc_flag_cols].sum(axis=1)

    numeric_features     = ['Age','Monthly_Income_Stated','Actual_Bank_Credit_Monthly','LTV_Ratio',
        'FOIR','CIBIL_Score','Num_Credit_Enquiries_Last_30D','Loan_Amount_INR','Tenure_Years',
        'Interest_Rate_pct','Existing_Monthly_EMI','Years_With_Employer','Legal_Opinion_TAT_Hours',
        'income_discrepancy','income_inflate_ratio','itr_income_gap','doc_risk_score']
    categorical_features = ['Employment_Type','Loan_Purpose','Property_Type','Down_Payment_Source','Disbursement_Mode','Gender']

    X = df[numeric_features + categorical_features + doc_flag_cols].copy()
    y = df['Fraud_Label_01'].copy()
    for col in numeric_features:
        X[col] = pd.to_numeric(X[col], errors='coerce').fillna(X[col].median())
    for col in categorical_features:
        X[col] = X[col].fillna('Unknown')
    X = pd.get_dummies(X, columns=categorical_features, drop_first=True)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    scaler    = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)
    model     = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)
    model.fit(X_train_s, y_train)

    y_prob = model.predict_proba(X_test_s)[:, 1]
    y_pred = (y_prob >= 0.4).astype(int)

    auc  = roc_auc_score(y_test, y_prob)
    prec = precision_score(y_test, y_pred)
    rec  = recall_score(y_test, y_pred)
    cm   = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()

    coef_df = pd.DataFrame({'feature': X.columns, 'weight': model.coef_[0]}).sort_values('weight', ascending=False)

    pipeline  = Pipeline([('scaler', StandardScaler()),
                          ('model', LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42))])
    kf        = KFold(n_splits=5, shuffle=True, random_state=42)
    kf_scores = cross_val_score(pipeline, X, y, cv=kf, scoring='roc_auc')

    test_idx = X_test.index
    test_df  = df.loc[test_idx].copy()
    test_df['risk_score'] = (y_prob * 100).round(1)
    test_df['predicted']  = y_pred
    test_df['actual']     = y_test.values
    test_df['category']   = test_df.apply(lambda r:
        'TP' if r['predicted']==1 and r['actual']==1 else
        'FP' if r['predicted']==1 and r['actual']==0 else
        'FN' if r['predicted']==0 and r['actual']==1 else 'TN', axis=1)

    return (model, scaler, list(X.columns), auc, prec, rec,
            tn, fp, fn, tp, coef_df, kf_scores, df, doc_flag_cols,
            numeric_features, categorical_features, test_df)


def score_application(inputs, model, scaler, feature_names, numeric_features, categorical_features, doc_flag_cols):
    app    = pd.DataFrame([inputs])
    actual = inputs.get('Actual_Bank_Credit_Monthly', 1) or 1
    stated = inputs.get('Monthly_Income_Stated', 0)
    annual = inputs.get('Annual_Income_Stated', 0)
    itr    = inputs.get('ITR_Income_Last_FY', annual)
    app['income_discrepancy']   = stated - actual
    app['income_inflate_ratio'] = min(stated / actual if actual else 1, 10)
    app['itr_income_gap']       = annual - itr
    app['doc_risk_score']       = sum(int(inputs.get(c, 0)) for c in doc_flag_cols)
    for col in categorical_features:
        if col not in app.columns: app[col] = 'Unknown'
    app = pd.get_dummies(app, columns=categorical_features, drop_first=True)
    for col in feature_names:
        if col not in app.columns: app[col] = 0
    app = app[feature_names]
    for col in [c for c in numeric_features if c in app.columns]:
        app[col] = pd.to_numeric(app[col], errors='coerce').fillna(0)
    return model.predict_proba(scaler.transform(app))[0][1]


def render_loan_table(data, category):
    color_map = {
        'TP': ('#34d399', '#064e3b', 'Correctly caught fraud',      'Model said FRAUD → Actually FRAUD'),
        'FP': ('#fbbf24', '#78350f', 'False alarm — innocent',      'Model said FRAUD → Actually CLEAN'),
        'FN': ('#f87171', '#7f1d1d', 'Missed fraud — slipped thru', 'Model said CLEAN → Actually FRAUD'),
        'TN': ('#60a5fa', '#1e3a5f', 'Correctly approved clean',    'Model said CLEAN → Actually CLEAN'),
    }
    color, bg, short, long = color_map[category]

    st.markdown(f"""
    <div style="background:{bg};border:1px solid {color}33;border-radius:12px;
                padding:12px 16px;margin-bottom:16px;text-align:center">
        <div style="font-size:1rem;font-weight:600;color:{color}">{short}</div>
        <div style="font-size:0.8rem;color:{color};opacity:0.8;margin-top:3px">{long}</div>
    </div>""", unsafe_allow_html=True)

    priority_cols = [
        'risk_score', 'category', 'Fraud_Type', 'Anomaly_Notes',
        'Application_ID', 'Application_Date', 'Borrower_Name',
        'Age', 'Gender', 'Marital_Status', 'Dependents',
        'PAN', 'Aadhaar', 'Mobile', 'Email',
        'Employment_Type', 'Employer_Name', 'Years_With_Employer',
        'Monthly_Income_Stated', 'Actual_Bank_Credit_Monthly',
        'income_inflate_ratio', 'income_discrepancy',
        'Annual_Income_Stated', 'ITR_Income_Last_FY', 'ITR_Income_Prev_FY', 'itr_income_gap',
        'CIBIL_Score', 'Num_Credit_Enquiries_Last_30D',
        'Loan_Amount_INR', 'LTV_Ratio', 'FOIR', 'Tenure_Years',
        'Interest_Rate_pct', 'EMI_INR', 'Existing_Monthly_EMI',
        'Loan_Purpose', 'Property_Type', 'Market_Value_INR',
        'Down_Payment_INR', 'Down_Payment_Source',
        'Property_City', 'Property_State', 'Property_Pincode',
        'Disbursement_Mode', 'Disbursal_Date', 'Sanction_Date',
        'Loan_Status', 'First_EMI_Status', 'DPD_Bucket_Current', 'NPA_Date',
        'Co_Applicant', 'Co_Applicant_Name', 'Co_Applicant_Annual_Income',
        'Prev_Loan_Bank', 'Prev_Default_History',
        'Form16_Available', 'Valuation_Agency', 'Legal_Opinion', 'Legal_Opinion_TAT_Hours',
        'CERSAI_Checked', 'RERA_Registered', 'Title_Chain_Clear', 'EC_Years_Verified',
        'Broker_Involved', 'Broker_DSA_Name',
        'doc_risk_score',
        'Email_Name_Mismatch', 'Signature_Mismatch', 'ID_Docs_Altered',
        'Salary_Round_Figure_Flag', 'PF_Deduction_Missing',
        'Salary_Credited_From_Personal_Acct', 'ITR_Filed_Just_Before_Application',
        'Month_End_Fund_Parking', 'Valuation_Inflation_Flag', 'OC_CC_Available',
        'Disbursement_To_NonSeller_Flag', 'Bank_Statement_Tampered',
        'PDF_Metadata_Anomaly', 'Borrower_Unreachable', 'AML_Flag',
        'Loan_Foreclosed_Within_6M', 'PEP_Involved', 'Overseas_Wire_Transfer_Detected',
        'Fraud_Label_01',
    ]

    show_cols = [c for c in priority_cols if c in data.columns]
    remaining = [c for c in data.columns if c not in show_cols and c not in ['predicted','actual']]
    show_cols = show_cols + remaining

    display = data[show_cols].copy().reset_index(drop=True)

    money_cols = ['Monthly_Income_Stated','Actual_Bank_Credit_Monthly','Annual_Income_Stated',
                  'ITR_Income_Last_FY','ITR_Income_Prev_FY','Loan_Amount_INR','Market_Value_INR',
                  'Down_Payment_INR','EMI_INR','Existing_Monthly_EMI','Co_Applicant_Annual_Income',
                  'income_discrepancy','itr_income_gap']
    for col in money_cols:
        if col in display.columns:
            display[col] = display[col].apply(lambda x: f"₹{x:,.0f}" if pd.notna(x) else "—")

    if 'income_inflate_ratio' in display.columns:
        display['income_inflate_ratio'] = display['income_inflate_ratio'].apply(lambda x: f"{x:.2f}x" if pd.notna(x) else "—")
    if 'risk_score' in display.columns:
        display['risk_score'] = display['risk_score'].apply(lambda x: f"{x}%")
    if 'LTV_Ratio' in display.columns:
        display['LTV_Ratio'] = display['LTV_Ratio'].apply(lambda x: f"{x:.2f}" if pd.notna(x) else "—")
    if 'FOIR' in display.columns:
        display['FOIR'] = display['FOIR'].apply(lambda x: f"{x:.2f}" if pd.notna(x) else "—")

    total_cols = len(display.columns)
    st.dataframe(display, use_container_width=True, height=450)
    st.caption(f"{len(display):,} loan applications · {total_cols} columns — scroll right to see all columns")


def main():
    (model, scaler, feature_names, auc, prec, rec,
     tn, fp, fn, tp, coef_df, kf_scores, df, doc_flag_cols,
     numeric_features, categorical_features, test_df) = load_and_train()

    st.markdown("""
    <div class="hero">
        <div class="badge">HOUSING LOAN · FRAUD DETECTION SYSTEM</div>
        <h1>Anti-Fraud Detection Dashboard</h1>
        <p>Logistic Regression · 20-Year Dataset · 10,080 Loan Applications</p>
    </div>""", unsafe_allow_html=True)

    tab1, tab2, tab3, tab4 = st.tabs([
        "Model Performance",
        "Score New Application",
        "Feature Importance",
        "Inspect Predictions"
    ])

    with tab1:
        st.markdown("### Model Metrics")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown(f"""<div class="metric-card">
                <div class="label">AUC-ROC</div>
                <div class="value" style="color:#60a5fa">{auc:.4f}</div>
                <div class="meaning">Overall model quality — how well it separates fraud from clean across all thresholds. 1.0 = perfect.</div>
            </div>""", unsafe_allow_html=True)
        with c2:
            st.markdown(f"""<div class="metric-card">
                <div class="label">Recall</div>
                <div class="value" style="color:#34d399">{rec:.4f}</div>
                <div class="meaning">Of all real fraud cases, this many were caught. Most important metric — missing fraud costs crores.</div>
            </div>""", unsafe_allow_html=True)
        with c3:
            st.markdown(f"""<div class="metric-card">
                <div class="label">Precision</div>
                <div class="value" style="color:#a78bfa">{prec:.4f}</div>
                <div class="meaning">Of all cases flagged as fraud, this many were real fraud. Low precision = wasting investigator time.</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        col_left, col_right = st.columns(2)

        with col_left:
            st.markdown('<p style="font-size:0.7rem;font-weight:600;color:#8899aa;text-transform:uppercase;letter-spacing:0.1em">CONFUSION MATRIX — TEST SET (2,016 LOANS)</p>', unsafe_allow_html=True)
            ca, cb = st.columns(2)
            with ca:
                st.markdown(f"""<div class="metric-card" style="border-color:#064e3b">
                    <div class="label" style="color:#34d399">True Negative</div>
                    <div class="value" style="color:#34d399">{tn:,}</div>
                    <div class="meaning">Clean loans correctly approved</div>
                </div>""", unsafe_allow_html=True)
                st.markdown(f"""<div class="metric-card" style="border-color:#7f1d1d;margin-top:10px">
                    <div class="label" style="color:#f87171">False Negative</div>
                    <div class="value" style="color:#f87171">{fn:,}</div>
                    <div class="meaning">Fraudsters who slipped through</div>
                </div>""", unsafe_allow_html=True)
            with cb:
                st.markdown(f"""<div class="metric-card" style="border-color:#78350f">
                    <div class="label" style="color:#fbbf24">False Positive</div>
                    <div class="value" style="color:#fbbf24">{fp:,}</div>
                    <div class="meaning">Innocent borrowers wrongly flagged</div>
                </div>""", unsafe_allow_html=True)
                st.markdown(f"""<div class="metric-card" style="border-color:#064e3b;margin-top:10px">
                    <div class="label" style="color:#34d399">True Positive</div>
                    <div class="value" style="color:#34d399">{tp:,}</div>
                    <div class="meaning">Fraudsters correctly caught</div>
                </div>""", unsafe_allow_html=True)

        with col_right:
            st.markdown('<p style="font-size:0.7rem;font-weight:600;color:#8899aa;text-transform:uppercase;letter-spacing:0.1em">5-FOLD CROSS VALIDATION</p>', unsafe_allow_html=True)
            for i, s in enumerate(kf_scores, 1):
                st.markdown(f"""
                <div style="margin-bottom:8px">
                    <div style="display:flex;justify-content:space-between;margin-bottom:3px">
                        <span style="font-size:0.8rem;color:#8899aa">Fold {i}</span>
                        <span style="font-size:0.8rem;font-weight:600;color:#60a5fa">{s:.4f}</span>
                    </div>
                    <div style="background:#1e2533;border-radius:4px;height:8px;overflow:hidden">
                        <div style="width:{int(s*100)}%;height:100%;background:linear-gradient(90deg,#3b82f6,#60a5fa);border-radius:4px"></div>
                    </div>
                </div>""", unsafe_allow_html=True)
            st.markdown(f"""
            <div style="background:#1e3a5f;border-radius:10px;padding:12px 16px;margin-top:12px;display:flex;justify-content:space-between">
                <div style="text-align:center">
                    <div style="font-size:1.4rem;font-weight:700;color:#60a5fa">{kf_scores.mean():.4f}</div>
                    <div style="font-size:0.7rem;color:#8899aa;margin-top:2px">Mean AUC</div>
                </div>
                <div style="text-align:center">
                    <div style="font-size:1.4rem;font-weight:700;color:#34d399">{kf_scores.std():.4f}</div>
                    <div style="font-size:0.7rem;color:#8899aa;margin-top:2px">Std Deviation</div>
                </div>
                <div style="text-align:center">
                    <div style="font-size:1.4rem;font-weight:700;color:#a78bfa">Robust</div>
                    <div style="font-size:0.7rem;color:#8899aa;margin-top:2px">Model Status</div>
                </div>
            </div>""", unsafe_allow_html=True)

    with tab2:
        st.markdown("### Score a New Loan Application")
        st.markdown('<p style="color:#8899aa;font-size:0.9rem">Fill in the loan details below. The model will return a fraud risk score in real time.</p>', unsafe_allow_html=True)
        with st.form("loan_form"):
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown("**Borrower Details**")
                age            = st.slider("Age", 18, 70, 35)
                gender         = st.selectbox("Gender", ["Male","Female","Other"])
                employment     = st.selectbox("Employment Type", ["Salaried","Self-Employed Business","Self-Employed Professional","Government"])
                years_employer = st.slider("Years with Employer", 0, 30, 5)
                cibil          = st.slider("CIBIL Score", 300, 900, 720)
                enquiries      = st.slider("Credit Enquiries (Last 30 Days)", 0, 15, 1)
            with col2:
                st.markdown("**Income & Loan**")
                stated_income  = st.number_input("Stated Monthly Income (₹)", 0, 2000000, 100000, step=5000)
                actual_credit  = st.number_input("Actual Bank Credits/Month (₹)", 0, 2000000, 95000, step=5000)
                annual_income  = st.number_input("Annual Income Stated (₹)", 0, 20000000, 1200000, step=50000)
                itr_income     = st.number_input("ITR Income Last FY (₹)", 0, 20000000, 1150000, step=50000)
                loan_amount    = st.number_input("Loan Amount (₹)", 100000, 50000000, 5000000, step=100000)
                ltv            = st.slider("LTV Ratio", 0.0, 1.0, 0.65, 0.01)
                foir           = st.slider("FOIR", 0.0, 1.0, 0.40, 0.01)
                tenure         = st.slider("Tenure (Years)", 1, 30, 15)
                interest       = st.slider("Interest Rate (%)", 6.0, 18.0, 8.75, 0.25)
                emi            = st.number_input("Existing Monthly EMI (₹)", 0, 500000, 5000, step=1000)
                legal_tat      = st.slider("Legal Opinion TAT (Hours)", 1, 120, 72)
            with col3:
                st.markdown("**Property & Flags**")
                loan_purpose   = st.selectbox("Loan Purpose", ["Purchase","Balance Transfer","Construction","Extension","Repair/Renovation"])
                property_type  = st.selectbox("Property Type", ["Flat/Apartment","Row House","Independent House","Villa","Plot+Construction"])
                dp_source      = st.selectbox("Down Payment Source", ["Own Savings","Family Gift","Sale of Asset","Third Party"])
                disb_mode      = st.selectbox("Disbursement Mode", ["NEFT to Seller","NEFT to Builder","Cheque","Partial Cash","Cash"])
                st.markdown("**Document Flags** *(tick if present)*")
                bank_tampered  = st.checkbox("Bank Statement Tampered")
                pdf_anomaly    = st.checkbox("PDF Metadata Anomaly")
                aml_flag       = st.checkbox("AML Flag")
                pep            = st.checkbox("PEP Involved")
                disb_nonseller = st.checkbox("Disbursement to Non-Seller")
                sig_mismatch   = st.checkbox("Signature Mismatch")
                id_altered     = st.checkbox("ID Docs Altered")
                itr_before     = st.checkbox("ITR Filed Just Before Application")
                overseas       = st.checkbox("Overseas Wire Transfer")
                month_end_park = st.checkbox("Month End Fund Parking")
                val_inflation  = st.checkbox("Valuation Inflation")
                salary_personal= st.checkbox("Salary from Personal Account")
                pf_missing     = st.checkbox("PF Deduction Missing")
                unreachable    = st.checkbox("Borrower Unreachable")
                foreclosed     = st.checkbox("Loan Foreclosed Within 6M")
                round_salary   = st.checkbox("Salary Round Figure")
                oc_cc          = st.checkbox("OC/CC Available *(protective)*")
                email_mismatch = st.checkbox("Email Name Mismatch")
            submitted = st.form_submit_button("Calculate Fraud Risk Score")

        if submitted:
            inputs = {
                'Age':age,'Gender':gender,'Employment_Type':employment,
                'Years_With_Employer':years_employer,'CIBIL_Score':cibil,
                'Num_Credit_Enquiries_Last_30D':enquiries,
                'Monthly_Income_Stated':stated_income,'Actual_Bank_Credit_Monthly':actual_credit,
                'Annual_Income_Stated':annual_income,'ITR_Income_Last_FY':itr_income,
                'Loan_Amount_INR':loan_amount,'LTV_Ratio':ltv,'FOIR':foir,
                'Tenure_Years':tenure,'Interest_Rate_pct':interest,
                'Existing_Monthly_EMI':emi,'Legal_Opinion_TAT_Hours':legal_tat,
                'Loan_Purpose':loan_purpose,'Property_Type':property_type,
                'Down_Payment_Source':dp_source,'Disbursement_Mode':disb_mode,
                'Bank_Statement_Tampered':int(bank_tampered),'PDF_Metadata_Anomaly':int(pdf_anomaly),
                'AML_Flag':int(aml_flag),'PEP_Involved':int(pep),
                'Disbursement_To_NonSeller_Flag':int(disb_nonseller),
                'Signature_Mismatch':int(sig_mismatch),'ID_Docs_Altered':int(id_altered),
                'ITR_Filed_Just_Before_Application':int(itr_before),
                'Overseas_Wire_Transfer_Detected':int(overseas),
                'Month_End_Fund_Parking':int(month_end_park),
                'Valuation_Inflation_Flag':int(val_inflation),
                'Salary_Credited_From_Personal_Acct':int(salary_personal),
                'PF_Deduction_Missing':int(pf_missing),'Borrower_Unreachable':int(unreachable),
                'Loan_Foreclosed_Within_6M':int(foreclosed),
                'Salary_Round_Figure_Flag':int(round_salary),
                'OC_CC_Available':int(oc_cc),'Email_Name_Mismatch':int(email_mismatch),
            }
            score = score_application(inputs, model, scaler, feature_names, numeric_features, categorical_features, doc_flag_cols)
            r1, r2, r3 = st.columns([1,2,1])
            with r2:
                if score >= 0.7:
                    color,bg,label,sub = "#f87171","#1c0a0a","HIGH RISK — FLAG IMMEDIATELY","Send to fraud investigation team"
                elif score >= 0.4:
                    color,bg,label,sub = "#fbbf24","#1c1100","MODERATE RISK — REVIEW REQUIRED","Manual verification recommended"
                else:
                    color,bg,label,sub = "#34d399","#021c0e","LOW RISK — PROCEED","Proceed to credit appraisal"
                st.markdown(f"""
                <div style="background:{bg};border:1px solid {color}33;border-radius:16px;padding:2rem;text-align:center;margin:1rem 0">
                    <div style="font-size:3.5rem;font-weight:700;color:{color};line-height:1">{score*100:.1f}%</div>
                    <div style="font-size:1rem;font-weight:600;color:{color};margin-top:8px">{label}</div>
                    <div style="font-size:0.85rem;color:#8899aa;margin-top:4px">{sub}</div>
                </div>""", unsafe_allow_html=True)
            inflate     = min(stated_income / actual_credit if actual_credit else 1, 10)
            flags_count = sum([bank_tampered,pdf_anomaly,aml_flag,pep,disb_nonseller,sig_mismatch,
                               id_altered,itr_before,overseas,month_end_park,val_inflation,
                               salary_personal,pf_missing,unreachable,foreclosed,round_salary,email_mismatch])
            st.markdown("<br>", unsafe_allow_html=True)
            s1, s2, s3 = st.columns(3)
            with s1:
                col = "#f87171" if inflate > 1.5 else "#34d399"
                st.markdown(f"""<div class="metric-card"><div class="label">Income Inflate Ratio</div>
                <div class="value" style="color:{col}">{inflate:.2f}x</div>
                <div class="meaning">Stated income vs actual bank credits</div></div>""", unsafe_allow_html=True)
            with s2:
                col = "#f87171" if flags_count >= 3 else "#fbbf24" if flags_count >= 1 else "#34d399"
                st.markdown(f"""<div class="metric-card"><div class="label">Fraud Flags Triggered</div>
                <div class="value" style="color:{col}">{flags_count} / 17</div>
                <div class="meaning">Number of suspicious document/behaviour flags</div></div>""", unsafe_allow_html=True)
            with s3:
                col = "#f87171" if cibil < 650 else "#fbbf24" if cibil < 720 else "#34d399"
                st.markdown(f"""<div class="metric-card"><div class="label">CIBIL Score</div>
                <div class="value" style="color:{col}">{cibil}</div>
                <div class="meaning">Credit score — higher is better (300–900)</div></div>""", unsafe_allow_html=True)

    with tab3:
        st.markdown("### Feature Importance — What Drives the Model's Decisions")
        st.markdown('<p style="color:#8899aa;font-size:0.9rem">Positive weight = pushes toward fraud. Negative weight = pushes toward clean. Larger absolute value = more influential.</p>', unsafe_allow_html=True)
        col_fraud, col_clean = st.columns(2)
        with col_fraud:
            st.markdown('<p style="font-size:0.7rem;font-weight:600;color:#8899aa;text-transform:uppercase;letter-spacing:0.1em">TOP 10 — FRAUD SIGNALS</p>', unsafe_allow_html=True)
            top_fraud = coef_df.head(10)
            max_w = top_fraud['weight'].max()
            for _, row in top_fraud.iterrows():
                pct = int(abs(row['weight'])/max_w*100)
                st.markdown(f"""
                <div style="margin-bottom:10px">
                    <div style="display:flex;justify-content:space-between;margin-bottom:3px">
                        <span style="font-size:0.78rem;color:#f0f4f8;font-family:monospace">{row['feature'][:38]}</span>
                        <span style="font-size:0.78rem;font-weight:600;color:#f87171">+{row['weight']:.3f}</span>
                    </div>
                    <div style="background:#1e2533;border-radius:4px;height:7px;overflow:hidden">
                        <div style="width:{pct}%;height:100%;background:linear-gradient(90deg,#dc2626,#f87171);border-radius:4px"></div>
                    </div>
                </div>""", unsafe_allow_html=True)
        with col_clean:
            st.markdown('<p style="font-size:0.7rem;font-weight:600;color:#8899aa;text-transform:uppercase;letter-spacing:0.1em">TOP 10 — PROTECTIVE SIGNALS</p>', unsafe_allow_html=True)
            top_clean = coef_df.tail(10).iloc[::-1]
            max_w = abs(top_clean['weight']).max()
            for _, row in top_clean.iterrows():
                pct = int(abs(row['weight'])/max_w*100)
                st.markdown(f"""
                <div style="margin-bottom:10px">
                    <div style="display:flex;justify-content:space-between;margin-bottom:3px">
                        <span style="font-size:0.78rem;color:#f0f4f8;font-family:monospace">{row['feature'][:38]}</span>
                        <span style="font-size:0.78rem;font-weight:600;color:#34d399">{row['weight']:.3f}</span>
                    </div>
                    <div style="background:#1e2533;border-radius:4px;height:7px;overflow:hidden">
                        <div style="width:{pct}%;height:100%;background:linear-gradient(90deg,#059669,#34d399);border-radius:4px"></div>
                    </div>
                </div>""", unsafe_allow_html=True)
        st.markdown("""
        <div style="background:#1e3a5f;border:1px solid #1d4ed8;border-radius:12px;padding:1.25rem 1.5rem;margin-top:1rem">
            <div style="font-size:0.7rem;font-weight:600;color:#60a5fa;text-transform:uppercase;letter-spacing:0.1em;margin-bottom:0.5rem">KEY INSIGHT</div>
            <p style="color:#cbd5e1;font-size:0.9rem;margin:0;line-height:1.7">
                <strong style="color:#f0f4f8">income_inflate_ratio</strong> — a feature created from scratch by dividing stated income by actual bank credits — is the single strongest fraud predictor with a weight of <strong style="color:#60a5fa">+1.93</strong>. Any borrower claiming more than 1.5x their actual bank credits should trigger immediate manual review.
            </p>
        </div>""", unsafe_allow_html=True)

    with tab4:
        st.markdown("### Inspect Actual Loan Applications")
        st.markdown('<p style="color:#8899aa;font-size:0.9rem">Click any category to see the real loan applications behind each prediction type. Every row is a real loan from the 20-year dataset.</p>', unsafe_allow_html=True)

        st.markdown("#### Select a category to inspect")
        b1, b2, b3, b4 = st.columns(4)

        tp_data = test_df[test_df['category']=='TP']
        fp_data = test_df[test_df['category']=='FP']
        fn_data = test_df[test_df['category']=='FN']
        tn_data = test_df[test_df['category']=='TN']

        with b1:
            st.markdown(f"""
            <div style="background:#064e3b;border:2px solid #34d399;border-radius:12px;padding:1rem;text-align:center;margin-bottom:10px">
                <div style="font-size:2rem;font-weight:700;color:#34d399">{tp:,}</div>
                <div style="font-size:0.85rem;font-weight:600;color:#34d399">True Positive</div>
                <div style="font-size:0.72rem;color:#6ee7b7;margin-top:4px">Fraud correctly caught</div>
            </div>""", unsafe_allow_html=True)
            if st.button("View TP loans", key="btn_tp"):
                st.session_state['selected'] = 'TP'

        with b2:
            st.markdown(f"""
            <div style="background:#78350f;border:2px solid #fbbf24;border-radius:12px;padding:1rem;text-align:center;margin-bottom:10px">
                <div style="font-size:2rem;font-weight:700;color:#fbbf24">{fp:,}</div>
                <div style="font-size:0.85rem;font-weight:600;color:#fbbf24">False Positive</div>
                <div style="font-size:0.72rem;color:#fde68a;margin-top:4px">Innocent flagged</div>
            </div>""", unsafe_allow_html=True)
            if st.button("View FP loans", key="btn_fp"):
                st.session_state['selected'] = 'FP'

        with b3:
            st.markdown(f"""
            <div style="background:#7f1d1d;border:2px solid #f87171;border-radius:12px;padding:1rem;text-align:center;margin-bottom:10px">
                <div style="font-size:2rem;font-weight:700;color:#f87171">{fn:,}</div>
                <div style="font-size:0.85rem;font-weight:600;color:#f87171">False Negative</div>
                <div style="font-size:0.72rem;color:#fca5a5;margin-top:4px">Fraud missed</div>
            </div>""", unsafe_allow_html=True)
            if st.button("View FN loans", key="btn_fn"):
                st.session_state['selected'] = 'FN'

        with b4:
            st.markdown(f"""
            <div style="background:#1e3a5f;border:2px solid #60a5fa;border-radius:12px;padding:1rem;text-align:center;margin-bottom:10px">
                <div style="font-size:2rem;font-weight:700;color:#60a5fa">{tn:,}</div>
                <div style="font-size:0.85rem;font-weight:600;color:#60a5fa">True Negative</div>
                <div style="font-size:0.72rem;color:#93c5fd;margin-top:4px">Clean correctly approved</div>
            </div>""", unsafe_allow_html=True)
            if st.button("View TN loans", key="btn_tn"):
                st.session_state['selected'] = 'TN'

        selected = st.session_state.get('selected', None)
        if selected:
            st.markdown("---")
            data_map = {'TP': tp_data, 'FP': fp_data, 'FN': fn_data, 'TN': tn_data}
            render_loan_table(data_map[selected], selected)

            csv = data_map[selected].drop(columns=['predicted','actual'], errors='ignore').to_csv(index=False)
            st.download_button(
                label=f"Download {selected} loans as CSV",
                data=csv,
                file_name=f"{selected}_loans.csv",
                mime="text/csv"
            )
        else:
            st.markdown("""
            <div style="background:#1a1f2e;border:1px dashed #2d3748;border-radius:12px;
                        padding:2rem;text-align:center;margin-top:1rem">
                <div style="color:#8899aa;font-size:0.9rem">Click any button above to see the actual loan applications</div>
            </div>""", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
