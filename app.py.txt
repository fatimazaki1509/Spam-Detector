import streamlit as st
import joblib

# Load your trained model
model = joblib.load('spam_classifier.pkl')  # Make sure this file exists

# Function to predict spam
def predict_spam(email_text):
    prediction = model.predict([email_text])
    return "Spam" if prediction[0] == 1 else "Not Spam"

# Streamlit UI
st.title("📧 Spam Email Classifier")
st.write("Enter an email below to check if it's spam or not.")

# Input text box
email_text = st.text_area("Enter email content:", "")

# Predict button
if st.button("Check Spam"):
    if email_text.strip():
        result = predict_spam(email_text)
        st.success(f"Prediction: **{result}**")
    else:
        st.warning("Please enter an email text to classify.")
