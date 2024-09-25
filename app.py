# the columns index
# ['bed', 'bath', 'house_size']

import streamlit as st
import numpy as np
import joblib as jl

scaler = jl.load("Scaler.pkl")

model = jl.load("model.pkl")

st.title("Real Estate Price Prediction App")

st.divider()

bed = st.number_input("Enter the number of bedrooms", value = 2, step = 1)
bath = st.number_input("Enter the number of bathrooms", value = 1, step = 1)
size = st.number_input("Enter the size", value = 1000, step = 50)

x = [bed, bath, size]

st.divider()

predict_button = st.button("Predict!")
st.divider()

if predict_button:
    st.balloons()
    x1 = np.array(x)
    x_array = scaler.transform([x1])

    prediction = model.predict(x_array)[0]

    st.write(f"The prediction is {prediction:.2f}")

else:
    "Please use the butrton to prediction"

