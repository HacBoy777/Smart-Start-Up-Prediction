
## THIS FILE WILL BE OUR ACTUAL FRONTEND

import streamlit as st
import pandas as pd
import pickle
import json

from Tabs.prediction_tab import show_prediction
from Tabs.analytic_tab import show_analytics
from Tabs.downloads_tab import show_downloads
from Pages.auth_page import show_auth_page

# SESSION STATE
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if "username" not in st.session_state:
    st.session_state.username = ""

# PAGE CONFIG
st.set_page_config(layout="wide")

st.markdown(
    "<h1 style='text-align: center;'>Startup Success Predictor App</h1>",
    unsafe_allow_html=True
)

# LOAD MODELS
def load_models():
    log_model = pickle.load(open("Models/logistic_model.pkl", "rb"))
    knn_model = pickle.load(open("Models/knn_model.pkl", "rb"))
    return log_model, knn_model
log_model, knn_model = load_models()

# LOAD DATA
def load_data():
    return pd.read_csv("Data/startup_dataset.csv")

df = load_data()

# LOAD METRICS
def load_metrics():

    with open("Models/logistic_metrics.json") as f:
        log_metrics = json.load(f)

    with open("Models/knn_metrics.json") as f:
        knn_metrics = json.load(f)

    with open("Models/random_forest_metrics.json") as f:
        rf_metrics = json.load(f)

    with open("Models/svm_metrics.json") as f:
        svm_metrics = json.load(f)

    return log_metrics, knn_metrics, rf_metrics, svm_metrics

log_metrics, knn_metrics, rf_metrics, svm_metrics = load_metrics()

# SHOW LOGIN PAGE FIRST
if not st.session_state.logged_in:
    show_auth_page()
    st.stop()

# SIDEBAR
st.sidebar.success(
    f"👤 Logged in as: {st.session_state.username}"
)

if st.sidebar.button("Logout"):
    st.session_state.logged_in = False
    st.session_state.username = ""
    st.rerun()

st.sidebar.header("Startup Parameters")

experience = st.sidebar.slider("Founder Experience", 0, 25, 5)
team = st.sidebar.slider("Team Size", 1, 60, 10)
funding = st.sidebar.slider("Funding (Million $)", 0.1, 50.0, 5.0)
market = st.sidebar.slider("Market Size", 1, 10, 5)
innovation = st.sidebar.slider("Innovation Score", 1, 10, 6)
marketing = st.sidebar.slider("Marketing Budget", 0.1, 28.0, 3.0)
competition = st.sidebar.slider("Competition Level", 1, 10, 5)
revenue = st.sidebar.slider("Revenue Growth", -10, 100, 20)

industry = 1
education = 1
stage = 2

input_data = pd.DataFrame([{
    "FounderExperience": experience,
    "TeamSize": team,
    "FundingAmount": funding,
    "MarketSize": market,
    "InnovationScore": innovation,
    "MarketingBudget": marketing,
    "CompetitionLevel": competition,
    "IndustryType": industry,
    "FounderEducation": education,
    "ProductStage": stage,
    "RevenueGrowth": revenue
}])

# MODEL PREDICTIONS
log_model = pickle.load(open("Models/logistic_model.pkl", "rb"))
prob_log = log_model.predict_proba(input_data)[0][1]

knn_model = pickle.load(open("Models/knn_model.pkl", "rb"))
prob_knn = knn_model.predict_proba(input_data)[0][1]

rf_model = pickle.load(open("Models/random_forest_model.pkl", "rb"))
prob_rf = rf_model.predict_proba(input_data)[0][1]

svm_model = pickle.load(open("Models/svm_model.pkl", "rb"))
prob_svm = svm_model.predict_proba(input_data)[0][1]

# DASHBOARD TABS
tab1, tab2, tab3 = st.tabs(
    ["Prediction", "Analytics", "Downloads"]
)

with tab1:
    show_prediction(
        prob_log,
        prob_knn,
        prob_rf,
        prob_svm,
        log_model,
        input_data,
        log_metrics,
        knn_metrics,
        rf_metrics,
        svm_metrics
    )

with tab2:
    show_analytics(
        df,
        knn_model,
        input_data
    )

with tab3:
    show_downloads(
        prob_log,
        input_data
    )

