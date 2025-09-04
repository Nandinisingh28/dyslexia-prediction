import streamlit as st
import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder

# Load saved components
with open('label_encoders.pkl', 'rb') as f:
    label_encoders = pickle.load(f)

with open('svm_model.pkl', 'rb') as f:
    model = pickle.load(f)

st.set_page_config(layout="centered")
# Streamlit user interface for prediction
st.title('Dyslexia Prediction Tool')


# Display the additional information below the title as plain text
st.subheader("Phonological Awareness Skills")
st.write("**DRW** - Difficulty in recognizing/reproducing rhyming words")
st.write("**DIS** - Difficulty in isolating sounds")
st.write("**DSIS** - Difficulty in segmenting individual sounds in words")

st.subheader("Alphabet")
st.write("**DLL** - Difficulty in learning letters")
st.write("**DLSL** - Difficulty in learning sounds of letters")

st.subheader("Decoding & Word Recognition")
st.write("**DUW** - Difficulty in sounding unfamiliar words")
st.write("**DRWI** - Difficulty in reading words in isolation")

st.subheader("Fluency")
st.write("**DRA** - Difficulty in reading accurately in context")
st.write("**DRGL** - Difficulty in reading grade level material")

st.subheader("Spelling")
st.write("**DMW** - Difficulty in memorizing words")
st.write("**DS** - Difficulty in spelling in context")

st.subheader("Comprehension")
st.write("**DRC** - Difficulty in reading comprehension")

# Define the 12 relevant features for input
features = ['DRW', 'DIS', 'DSIS', 'DLL', 'DLSL', 'DUW', 'DRWI', 'DRA', 'DRGL', 'DMW', 'DS', 'DRC']

# Create input fields for each relevant feature
input_data = {}
for feature in features:
    options = sorted(list(set(label_encoders[feature].classes_)))  # List unique class options for each feature
    selected_value = st.selectbox(f'Select value for {feature}:', options=options)
    input_data[feature] = selected_value


if st.button('Predict Dyslexia'):
    try:
        # Encode the input data
        encoded_data = [label_encoders[feature].transform([input_data[feature]])[0] for feature in features]

        # Ensure the input matches the expected number of features
        assert len(encoded_data) == len(features), "The input does not match the expected number of features."

        # Wrap the encoded data to form a 2D array
        input_data_2d = [encoded_data]  # This makes it a 2D array

        # Make prediction
        prediction = model.predict(input_data_2d)
        result = 'Has Dyslexia' if prediction[0] == 1 else 'No Dyslexia'
        st.write(f'The model predicts: **{result}**')
        
    except ValueError as e:
        st.error(f"Error: {e}")
    except AssertionError as e:
        st.error(f"Error: {e}")

