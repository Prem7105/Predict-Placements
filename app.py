import streamlit as st
import pickle
import numpy as np

# Load the saved model and scaler
try:
    with open('model.pkl', 'rb') as model_file:
        clf = pickle.load(model_file)
    with open('scaler.pkl', 'rb') as scaler_file:
        scaler = pickle.load(scaler_file)
except FileNotFoundError:
    st.error("Model or scaler file not found. Make sure 'model.pkl' and 'scaler.pkl' are in the same directory.")
    st.stop()


# App Title
st.title('Student Placement Predictor 🎓')
st.write("Enter a student's CGPA and IQ to predict their placement.")

# Input fields for user
cgpa = st.number_input('Enter CGPA', min_value=0.0, max_value=10.0, step=0.1, value=8.0)
iq = st.number_input('Enter IQ', min_value=50, max_value=200, step=1, value=100)

# Prediction button
if st.button('Predict Placement'):
    # 1. Create a numpy array from the inputs
    query_point = np.array([cgpa, iq]).reshape(1, -1)
    
    # 2. Scale the input data
    query_point_transformed = scaler.transform(query_point)
    
    # 3. Make a prediction
    prediction = clf.predict(query_point_transformed)
    
    # 4. Display the result
    if prediction[0] == 1:
        st.success('🎉 The student is likely to be placed!')
    else:
        st.error('😔 The student is not likely to be placed.')
