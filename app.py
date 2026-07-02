"""Streamlit web interface for the Adult Income classifier."""

from __future__ import annotations

import streamlit as st

from src.data import CATEGORICAL_COLUMNS
from src.train import load_model

st.set_page_config(page_title="Adult Income Classifier", page_icon="💰")

st.title("Adult Income Classifier")
st.markdown(
    "Predict whether an individual's annual income exceeds **$50K** based on census data."
)


def load():
    """Load the persisted model, training feature names, and category options."""
    model = load_model()
    preprocessor = model.named_steps["preprocessor"]
    categorical_options = {
        col: preprocessor.named_transformers_["cat"].categories_[i].tolist()
        for i, col in enumerate(CATEGORICAL_COLUMNS)
    }
    return model, categorical_options


@st.cache_resource
def cached_load():
    return load()


try:
    model, categorical_options = cached_load()
except FileNotFoundError:
    st.error(
        "Trained model not found. Run `uv run python -m src.main` to train and save the model."
    )
    st.stop()


with st.form("prediction_form"):
    st.subheader("Enter details")

    age = st.number_input("Age", min_value=17, max_value=90, value=30)
    education_num = st.number_input(
        "Years of education", min_value=1, max_value=16, value=10
    )
    capital_gain = st.number_input("Capital gain ($)", min_value=0, value=0, step=100)
    capital_loss = st.number_input("Capital loss ($)", min_value=0, value=0, step=100)
    hours_per_week = st.number_input(
        "Hours per week", min_value=1, max_value=99, value=40
    )

    workclass = st.selectbox("Workclass", categorical_options["workclass"])
    marital_status = st.selectbox(
        "Marital status", categorical_options["marital-status"]
    )
    occupation = st.selectbox("Occupation", categorical_options["occupation"])
    relationship = st.selectbox("Relationship", categorical_options["relationship"])
    race = st.selectbox("Race", categorical_options["race"])
    sex = st.selectbox("Sex", categorical_options["sex"])
    native_country = st.selectbox(
        "Native country", categorical_options["native-country"]
    )

    submitted = st.form_submit_button("Predict")

if submitted:
    input_data = {
        "age": age,
        "education-num": education_num,
        "capital-gain": capital_gain,
        "capital-loss": capital_loss,
        "hours-per-week": hours_per_week,
        "workclass": workclass,
        "marital-status": marital_status,
        "occupation": occupation,
        "relationship": relationship,
        "race": race,
        "sex": sex,
        "native-country": native_country,
    }

    import pandas as pd

    sample = pd.DataFrame([input_data])
    prediction = model.predict(sample)[0]
    probabilities = model.predict_proba(sample)[0]
    proba_dict = dict(zip(model.classes_, probabilities))

    st.subheader("Result")
    if prediction == ">50K":
        st.success(f"Predicted income: **{prediction}**")
    else:
        st.info(f"Predicted income: **{prediction}**")

    st.write("Class probabilities:")
    st.json({cls: f"{p:.3f}" for cls, p in proba_dict.items()})
