import os
import joblib
import pandas as pd
import streamlit as st
from huggingface_hub import hf_hub_download

st.set_page_config(page_title="Tourism Purchase Predictor", page_icon="✈️", layout="wide")

@st.cache_resource
def load_model():
    path = hf_hub_download(
        repo_id=os.environ["HF_MODEL_REPO"],
        filename="model.joblib",
        token=os.getenv("HF_TOKEN"),
    )
    return joblib.load(path)

st.title("✈️ Tourism Package Purchase Predictor")
st.write("Enter information available before customer contact.")
st.caption("The model was trained on historical general package purchases, not wellness-specific outcomes.")

with st.form("customer_form"):
    first, second = st.columns(2)
    with first:
        age = st.number_input("Age", 18, 100, 35)
        city_tier = st.selectbox("City Tier", [1, 2, 3])
        occupation = st.selectbox("Occupation", ["Salaried", "Freelancer", "Small Business", "Large Business"])
        gender = st.selectbox("Gender", ["Female", "Male"])
        persons = st.number_input("Number of Persons Visiting", 1, 10, 2)
        property_star = st.selectbox("Preferred Property Star", [3, 4, 5])
        marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced", "Unmarried"])
    with second:
        trips = st.number_input("Number of Trips per Year", 0, 30, 3)
        passport = st.selectbox("Has Passport", [0, 1], format_func=lambda x: "Yes" if x else "No")
        own_car = st.selectbox("Owns Car", [0, 1], format_func=lambda x: "Yes" if x else "No")
        children = st.number_input("Number of Children Visiting", 0, 10, 1)
        designation = st.selectbox("Designation", ["Executive", "Manager", "Senior Manager", "AVP", "VP"])
        income = st.number_input("Monthly Income", 0.0, value=25000.0, step=500.0)
    submitted = st.form_submit_button("Predict Purchase Likelihood", type="primary")

if submitted:
    input_data = pd.DataFrame([{
        "Age": age,
        "CityTier": city_tier,
        "Occupation": occupation,
        "Gender": gender,
        "NumberOfPersonVisiting": persons,
        "PreferredPropertyStar": property_star,
        "MaritalStatus": marital_status,
        "NumberOfTrips": trips,
        "Passport": passport,
        "OwnCar": own_car,
        "NumberOfChildrenVisiting": children,
        "Designation": designation,
        "MonthlyIncome": income,
    }])
    model = load_model()
    probability = float(model.predict_proba(input_data)[0, 1])
    prediction = int(probability >= 0.50)
    st.metric("Purchase score", f"{probability:.1%}")
    if prediction:
        st.success("This customer is classified as likely to purchase.")
    else:
        st.info("This customer is classified as less likely to purchase.")
