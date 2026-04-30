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
from sagemaker.deserializers import JSONDeserializer
 
from sklearn.pipeline import Pipeline
import shap
 
from joblib import load
 
# Setup & Path Configuration
warnings.simplefilter("ignore")
 
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
if project_root not in sys.path:
    sys.path.append(project_root)
 
file_path = os.path.join(project_root, 'Portfolio/X_train.csv')
 
dataset = pd.read_csv(file_path)
dataset = dataset.drop(['Unnamed: 0'], axis=1)
 
# AWS Secrets
aws_id = st.secrets["aws_credentials"]["AWS_ACCESS_KEY_ID"]
aws_secret = st.secrets["aws_credentials"]["AWS_SECRET_ACCESS_KEY"]
aws_token = st.secrets["aws_credentials"]["AWS_SESSION_TOKEN"]
aws_bucket = st.secrets["aws_credentials"]["AWS_BUCKET"]
aws_endpoint = st.secrets["aws_credentials"]["AWS_ENDPOINT"]
 
@st.cache_resource
def get_session(aws_id, aws_secret, aws_token):
    return boto3.Session(
        aws_access_key_id=aws_id,
        aws_secret_access_key=aws_secret,
        aws_session_token=aws_token,
        region_name='us-east-1'
    )
 
session = get_session(aws_id, aws_secret, aws_token)
sm_session = sagemaker.Session(boto_session=session)
 
MODEL_INFO = {
    "endpoint": aws_endpoint,
    "explainer": "shap_explainer.pkl",
    "pipeline": "finalized_loan_model.tar.gz",
    "keys": ['grade_encoded','term','debt_settlement_flag_Y','high_int_rate'],
    "inputs": [{"name": k, "type": "number", "min": -1.0, "max": 1.0, "default": 0.0, "step": 0.01} 
               for k in ['grade_encoded','term','debt_settlement_flag_Y','high_int_rate']]
}
 
def load_pipeline(_session, bucket, key):
    s3_client = _session.client('s3')
    filename = MODEL_INFO["pipeline"]
 
    s3_client.download_file(
        Filename=filename,
        Bucket=bucket,
        Key=f"{key}/{os.path.basename(filename)}"
    )
 
    with tarfile.open(filename, "r:gz") as tar:
        tar.extractall(path=".")
        joblib_file = [f for f in tar.getnames() if f.endswith('.pkl')][0]
 
    return joblib.load(f"{joblib_file}")
 
def load_shap_explainer(_session, bucket, key, local_path):
    s3_client = _session.client('s3')
 
    if not os.path.exists(local_path):
        s3_client.download_file(Filename=local_path, Bucket=bucket, Key=key)
       
    with open(local_path, "rb") as f:
        return load(f)
 
# ✅ FIXED CALL MODEL API
def call_model_api(input_df):
    predictor = Predictor(
        endpoint_name=MODEL_INFO["endpoint"],
        sagemaker_session=sm_session,
        serializer=None,
        deserializer=JSONDeserializer()
    )

    try:
        if isinstance(input_df, dict):
            input_df = pd.DataFrame([input_df])

        input_df = input_df.applymap(lambda x: x[0] if isinstance(x, dict) else x)

        payload = input_df.to_json(orient="records")

        raw_pred = predictor.predict(
            payload,
            initial_args={"ContentType": "application/json"}
        )

        pred_val = raw_pred[0] if isinstance(raw_pred, list) else raw_pred

        mapping = {0: "Good Loan", 1: "Bad Loan"}
        return mapping.get(pred_val), 200

    except Exception as e:
        return f"Error: {str(e)}", 500
 
def display_explanation(input_df, session, aws_bucket):
    explainer_name = MODEL_INFO["explainer"]
    explainer = load_shap_explainer(
        session,
        aws_bucket,
        posixpath.join('explainer', explainer_name),
        os.path.join(tempfile.gettempdir(), explainer_name)
    )
   
    best_pipeline = load_pipeline(session, aws_bucket, 'sklearn-pipeline-deployment')
    preprocessing_pipeline = Pipeline(steps=best_pipeline.steps[:-2])
 
    input_df = pd.DataFrame(input_df)
    input_df_transformed = preprocessing_pipeline.transform(input_df)
 
    dataset_1 = dataset.iloc[:, 0:]
    feature_names = dataset_1.columns[1:]
 
    selector = best_pipeline.named_steps['selector']
    selected_features = feature_names[selector.get_support()]
 
    input_df_transformed = pd.DataFrame(input_df_transformed, columns=selected_features)
 
    shap_values = explainer(input_df_transformed)
   
    st.subheader("🔍 Decision Transparency (SHAP)")
    fig, ax = plt.subplots(figsize=(10, 4))
    shap.plots.waterfall(shap_values[0, :, 1])
    st.pyplot(fig)
 
    top_feature = pd.Series(
        shap_values[0, :, 1].values,
        index=shap_values[0, :, 1].feature_names
    ).abs().idxmax()
 
    st.info(f"**Business Insight:** The most influential factor in this decision was **{top_feature}**.")
 
# UI
st.set_page_config(page_title="ML Deployment", layout="wide")
st.title("👨‍💻 ML Deployment")
 
with st.form("pred_form"):
    st.subheader("Inputs")
    cols = st.columns(2)
    user_inputs = {}
   
    for i, inp in enumerate(MODEL_INFO["inputs"]):
        with cols[i % 2]:
            user_inputs[inp['name']] = st.number_input(
                inp['name'].replace('_', ' ').upper(),
                min_value=inp['min'],
                max_value=inp['max'],
                value=inp['default'],
                step=inp['step']
            )
   
    submitted = st.form_submit_button("Run Prediction")
 
original = dataset.iloc[0:1].to_dict(orient='records')[0]
original.pop('Unnamed: 0', None)
original.update(user_inputs)
 
if submitted:
    res, status = call_model_api(original)
 
    if status == 200:
        st.metric("Prediction Result", res)
        display_explanation(original, session, aws_bucket)
    else:
        st.error(res)
