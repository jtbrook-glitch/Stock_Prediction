import os, sys, warnings
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import posixpath

import joblib
import tarfile
import tempfile

import boto3
import sagemaker
from sagemaker.predictor import Predictor
from sagemaker.serializers import NumpySerializer
from sagemaker.deserializers import NumpyDeserializer

import shap

# =========================
# Setup & Path Configuration
# =========================
warnings.simplefilter("ignore")

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.feature_utils import extract_features

# =========================
# Secrets
# =========================
aws_id = st.secrets["aws_credentials"]["AWS_ACCESS_KEY_ID"]
aws_secret = st.secrets["aws_credentials"]["AWS_SECRET_ACCESS_KEY"]
aws_token = st.secrets["aws_credentials"]["AWS_SESSION_TOKEN"]
aws_bucket = st.secrets["aws_credentials"]["AWS_BUCKET"]
aws_endpoint = st.secrets["aws_credentials"]["AWS_ENDPOINT"]

# =========================
# AWS Session
# =========================
@st.cache_resource
def get_session(_aws_id, _aws_secret, _aws_token):
    return boto3.Session(
        aws_access_key_id=_aws_id,
        aws_secret_access_key=_aws_secret,
        aws_session_token=_aws_token,
        region_name="us-east-1",
    )

session = get_session(aws_id, aws_secret, aws_token)
sm_session = sagemaker.Session(boto_session=session)

# =========================
# Data & Model Config
# =========================
df_features = extract_features()

MODEL_INFO = {
    "endpoint": aws_endpoint,
    "explainer": "explainer.shap",
    "pipeline": "finalized_model.tar.gz",

    # What the SageMaker endpoint expects (15 features)
    "keys": [
        "JPM","MS","C","WFC","BAC","COF",
        "DEXJPUS","DEXUSUK","SP500","DJIA","VIXCLS",
        "GS_mom5","GS_vol20","GS_hl_range","GS_ma10_50_gap"
    ],

    # What you want users to type in (11 features)
    "ui_keys": ["JPM","MS","C","WFC","BAC","COF","DEXJPUS","DEXUSUK","SP500","DJIA","VIXCLS"],

    "inputs": [
        {"name": k, "type": "number", "min": -1.0, "max": 1.0, "default": 0.0, "step": 0.01}
        for k in ["JPM","MS","C","WFC","BAC","COF","DEXJPUS","DEXUSUK","SP500","DJIA","VIXCLS"]
    ]
}

# =========================
# SHAP Explainer Loader
# =========================
def load_shap_explainer(_session, bucket, key, local_path):
    s3_client = _session.client("s3")

    # Only download if it doesn't exist locally
    if not os.path.exists(local_path):
        s3_client.download_file(Filename=local_path, Bucket=bucket, Key=key)

    with open(local_path, "rb") as f:
        return shap.Explainer.load(f)

# =========================
# Prediction Logic
# =========================
def call_model_api(input_df: pd.DataFrame):
    predictor = Predictor(
        endpoint_name=MODEL_INFO["endpoint"],
        sagemaker_session=sm_session,
        serializer=NumpySerializer(),
        deserializer=NumpyDeserializer(),
    )

    try:
        # NumpySerializer expects a numpy array
        x = input_df.to_numpy(dtype=np.float32)  # shape (1, 15)
        raw_pred = predictor.predict(x)

        # Make robust to shape (1,1), (1,), scalar, etc.
        pred_val = np.array(raw_pred).reshape(-1)[0]
        return round(float(pred_val), 4), 200
    except Exception as e:
        return f"Error: {str(e)}", 500

# =========================
# Local Explainability
# =========================
def display_explanation(input_df, _session, _aws_bucket):
    explainer_name = MODEL_INFO["explainer"]
    explainer_s3_key = posixpath.join("explainer", explainer_name)
    local_path = os.path.join(tempfile.gettempdir(), explainer_name)

    explainer = load_shap_explainer(_session, _aws_bucket, explainer_s3_key, local_path)

    shap_values = explainer(input_df)

    st.subheader("🔍 Decision Transparency (SHAP)")

    # Let SHAP draw, but prevent it from popping its own window
    shap.plots.waterfall(shap_values[0], max_display=10, show=False)
    st.pyplot(plt.gcf(), clear_figure=True)

    # top feature (best practice: use absolute value)
    try:
        vals = shap_values[0].values
        names = shap_values[0].feature_names
        top_idx = int(np.argmax(np.abs(vals)))
        top_feature = names[top_idx]
        st.info(f"**Business Insight:** The most influential factor in this decision was **{top_feature}**.")
    except Exception:
        pass

# =========================
# Streamlit UI
# =========================
st.set_page_config(page_title="ML Deployment", layout="wide")
st.title("👨‍💻 ML Deployment")

with st.form("pred_form"):
    st.subheader("Inputs")
    cols = st.columns(2)
    user_inputs = {}

    for i, inp in enumerate(MODEL_INFO["inputs"]):
        with cols[i % 2]:
            user_inputs[inp["name"]] = st.number_input(
                inp["name"].replace("_", " ").upper(),
                min_value=inp["min"],
                max_value=inp["max"],
                value=inp["default"],
                step=inp["step"],
            )

    submitted = st.form_submit_button("Run Prediction")

# =========================
# Run Prediction
# =========================
if submitted:
    # Start from last feature row
    row = df_features.iloc[-1].copy()

    # Overwrite only user-provided 11 inputs
    for k in MODEL_INFO["ui_keys"]:
        row[k] = user_inputs[k]

    # Align to the 15 expected model keys (fills missing engineered cols with 0.0)
    missing = [k for k in MODEL_INFO["keys"] if k not in row.index]
    if missing:
        st.warning(f"Missing engineered features in df_features (filled with 0.0): {missing}")

    row_aligned = row.reindex(MODEL_INFO["keys"], fill_value=0.0)
    input_df = pd.DataFrame([row_aligned], columns=MODEL_INFO["keys"])

    # Call endpoint
    res, status = call_model_api(input_df)

    if status == 200:
        st.metric("Prediction Result", res)
        display_explanation(input_df, session, aws_bucket)
    else:
        st.error(res)


















