import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib
import os

st.set_page_config(page_title="House Price Prediction App", layout="centered")
st.title("🏡 Real-Time House Price Predictor")

MODEL_PATH = "house_price_predictor.pkl"

# Step 1: Train and save model if not present
if not os.path.exists(MODEL_PATH):
    st.info("Training model since no .pkl file found...")

    # Simulate data
    np.random.seed(42)
    n_samples = 500
    df = pd.DataFrame({
        "GrLivArea": np.random.randint(800, 3000, n_samples),
        "OverallQual": np.random.randint(1, 11, n_samples),
        "YearBuilt": np.random.randint(1950, 2025, n_samples),
        "Neighborhood": np.random.choice([
            "CollgCr", "Veenker", "Crawfor", "NoRidge", "Mitchel", "Somerst",
            "NWAmes", "OldTown", "BrkSide", "Sawyer", "NridgHt", "NAmes"
        ], n_samples),
        "RecentPriceTrend": np.random.uniform(-5, 5, n_samples),
        "PropertyTax": np.random.randint(1000, 5000, n_samples),
        "CrimeRate": np.random.uniform(1, 10, n_samples),
        "SchoolRating": np.random.randint(1, 11, n_samples),
        "Price_per_sqft": np.random.randint(0, 2, n_samples),
        "Distance_to_city": np.random.uniform(0.5, 30, n_samples),
    })

    df["SalePrice"] = (
        df["GrLivArea"] * 120 +
        df["OverallQual"] * 5000 +
        (2025 - df["YearBuilt"]) * -150 +
        df["RecentPriceTrend"] * 1000 +
        np.random.normal(0, 10000, n_samples)
    )

    X = df.drop("SalePrice", axis=1)
    y = df["SalePrice"]

    # Pipeline
    categorical = ["Neighborhood"]
    numeric = [col for col in X.columns if col not in categorical]

    preprocessor = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical)
    ], remainder="passthrough")

    model = Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", RandomForestRegressor(n_estimators=100, random_state=42))
    ])

    model.fit(X, y)
    joblib.dump(model, MODEL_PATH)
    st.success("Model trained and saved!")

# Step 2: Load model
model = joblib.load(MODEL_PATH)

# Step 3: Input interface
st.markdown("### Enter House Details")

with st.form("predict_form"):
    col1, col2 = st.columns(2)

    with col1:
        gr_liv_area = st.number_input("Living Area (sq. ft)", value=1500)
        year_built = st.number_input("Year Built", min_value=1900, max_value=2025, value=2005)
        property_tax = st.number_input("Annual Property Tax (USD)", value=2500)
        crime_rate = st.slider("Crime Rate (1=Low, 10=High)", 1.0, 10.0, 5.0)

    with col2:
        overall_qual = st.selectbox("Overall Quality (1-10)", list(range(1, 11)), index=7)
        school_rating = st.slider("School Rating (1=Low, 10=High)", 1, 10, 7)
        distance_to_city = st.slider("Distance to City Center (km)", 0.5, 30.0, 5.0, step=0.5)
        neighborhood = st.selectbox("Neighborhood", [
            "CollgCr", "Veenker", "Crawfor", "NoRidge", "Mitchel", "Somerst",
            "NWAmes", "OldTown", "BrkSide", "Sawyer", "NridgHt", "NAmes"
        ])
    
    recent_trend = st.slider("Recent Price Trend (%)", -5.0, 5.0, 0.0, step=0.1)

    submitted = st.form_submit_button("Predict House Price")

# Step 4: Prediction
if submitted:
    input_df = pd.DataFrame({
        "GrLivArea": [gr_liv_area],
        "OverallQual": [overall_qual],
        "YearBuilt": [year_built],
        "Neighborhood": [neighborhood],
        "RecentPriceTrend": [recent_trend],
        "PropertyTax": [property_tax],
        "CrimeRate": [crime_rate],
        "SchoolRating": [school_rating],
        "Price_per_sqft": [1 if gr_liv_area else 0],
        "Distance_to_city": [distance_to_city]
    })

    prediction = model.predict(input_df)[0]
    st.success(f"Estimated House Price: ₹ {prediction:,.2f}")
    st.markdown("### Input Summary")
    st.dataframe(input_df)
