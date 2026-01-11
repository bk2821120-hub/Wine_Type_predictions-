# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import joblib
import streamlit as st

st.set_page_config(page_title="Wine type classifier")
st.title("Wine Type lassifier App")
st.write("Predict whether the wine is **red** or **white** using chemical properties")
fixed_acidity =st.number_input("Value of Fixed acidity" ,value=None)
volatile_acidity =st.number_input(" Value of volatile_acidity",value=None)
citric_acid =st.number_input(" Value of citric_acid",value=None)
residual_sugar =st.number_input(" Value of residual_sugar",value=None)
chlorides =st.number_input(" Value of chlorides",value=None)
free_sulfur_dioxide =st.number_input(" Value of free_sulfur_dioxide",value=None)
total_sulfur_dioxide =st.number_input(" Value of total_sulfur_dioxide",value=None)
density =st.number_input("Value of density",value=None)
pH =st.number_input("Value of pH",value=None)
sulphates=st.number_input("Value of sulphates",value=None)
alcohol =st.number_input("Value of alcohol",value=None)
quality=st.number_input("Value of quality",value=None)

if st.button("Predict"):
  input_data = np.array([[
      fixed_acidity,
      volatile_acidity,
      citric_acid,
      residual_sugar,
      chlorides,
      free_sulfur_dioxide,
      total_sulfur_dioxide,
      density,
      pH,
      sulphates,
      alcohol,
      quality

  ]])
  prediction = model.predict(input_data)[0]
  if prediction == 0:
    st.error("The wine is red")
  else:
    st.success("The wine is white")
