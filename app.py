# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import joblib
import streamlit as st

# Load the model
model = joblib.load('rfr_model.pkl')

# Streamlit page setup
st.set_page_config(page_title='Wine Type Classification', layout='centered')
st.title('Wine Type Classification App')
st.write("Predict whether the wine is **red** or **white** using chemical properties")

# Input fields with default numeric values
fixed_acidity = st.number_input('Value of fixed acidity', value=7.0)
volatile_acidity = st.number_input('Value of volatile acidity', value=0.7)
citric_acid = st.number_input('Value of citric acid', value=0.0)
residual_sugar = st.number_input('Value of residual sugar', value=2.0)
chlorides = st.number_input('Value of chlorides', value=0.08)
free_sulfur_dioxide = st.number_input('Value of free sulfur dioxide', value=15.0)
total_sulfur_dioxide = st.number_input('Value of total sulfur dioxide', value=46.0)
density = st.number_input('Value of density', value=0.997)
pH = st.number_input('Value of pH', value=3.2)
sulphates = st.number_input('Value of sulphates', value=0.5)
alcohol = st.number_input('Value of alcohol', value=10.0)
quality = st.number_input('Value of quality', value=5.0)

# Predict button
if st.button('Predict'):
    # Create a DataFrame with proper feature names
    input_data = pd.DataFrame([[
        fixed_acidity,
        volatile_acidity,
        citric_acid,
        residual_sugar,
        chlorides,
        free_sulfur_dioxide,
        total_sulfur_dioxide,
