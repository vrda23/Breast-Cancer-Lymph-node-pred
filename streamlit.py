import streamlit as st
import pandas as pd
import pickle
import numpy as np

# Load your model
model = pickle.load(open('model.pkl', 'rb'))

# Create a DataFrame to store the dummy encoded version of the hist_type feature
dummy_encoded_hist_type = pd.DataFrame({
    "NOS invasive": [1, 0, 0, 0, 0, 0, 0, 0, 0],
    "Lobular invasive": [0, 1, 0, 0, 0, 0, 0, 0, 0],
    "Metaplastic": [0, 0, 1, 0, 0, 0, 0, 0, 0],
    "Ca with medullary features": [0, 0, 0, 1, 0, 0, 0, 0, 0],
    "Mucinose invasive": [0, 0, 0, 0, 1, 0, 0, 0, 0],
    "Ca with apocrine differentiation": [0, 0, 0, 0, 0, 1, 0, 0, 0],
    "Other rare types": [0, 0, 0, 0, 0, 0, 1, 0, 0],
    "tubularni invazivni": [0, 0, 0, 0, 0, 0, 0, 1, 0],
    "micropapilary invasive": [0, 0, 0, 0, 0, 0, 0, 0, 1]
})


st.title("Breast Cancer Lymph Node Diagnosis")

# Get the user inputs for the features
age = st.number_input("Enter age")
tumour_size = st.number_input("Enter tumour size")
hist_type = st.selectbox("Select histological type:", ["NOS invasive","Lobular invasive","Metaplastic","Ca with medullary features","Mucinose invasive","Ca with apocrine differentiation","Other rare types",                                    "tubularni invazivni", "micropapilary invasive"])
ER = st.slider("ER score (1-100):",  min_value=0, max_value=100)
PR = st.slider("PR score (1-100):",  min_value=0, max_value=100)
Ki67 = st.slider("Ki-67 score (1-100):",  min_value=0, max_value=100)
tumour_grade = st.slider("Tumour grade (1-3):", 1, 3, 2)
her2 = st.slider("HER-2 score (0-3):", 0, 3, 1)

# Extract the corresponding row of binary values from the dummy encoded version of the hist_type feature
hist_type_encoded = dummy_encoded_hist_type.get(hist_type, [0,0,0,0,0,0,0,0,0]).values


# Combine the inputs into a single feature vector
input_vector = np.concatenate([hist_type_encoded, [age, tumour_size, ER, PR, Ki67, tumour_grade, her2]])

# Run your model on the input vector to make a prediction

if st.button("Get Your Prediction"):
    

    prediction = model.predict_proba([input_vector])[:,1]
    
    # Show the prediction to the user
    st.write("Prediction:", prediction)


