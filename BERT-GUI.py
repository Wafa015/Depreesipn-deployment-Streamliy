import streamlit as st
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

# Set page configuration
st.set_page_config(page_title="Depression Detection App", layout="wide")

# Load the pre-trained model and tokenizer
model_path = "depression_bert_model"
model = AutoModelForSequenceClassification.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

# Sidebar for navigation
page = st.sidebar.selectbox("Select Page", ["Home", "Settings"])

# Function to set styles based on mode
def set_styles(mode):
    if mode == "Dark Mode":
        st.markdown("""
        <style>
        .stApp {
            background-color: #333333;
            color: white;
        }
        h1, h3, p {
            color: white !important;
        }
        </style>
        """, unsafe_allow_html=True)
    elif mode == "Light Mode":
        st.markdown("""
        <style>
        .stApp {
            background-color: white;
            color: black;
        }
        h1, h3, p {
            color: black !important;
        }
        </style>
        """, unsafe_allow_html=True)
    else:  # System Mode
        st.markdown("""
        <style>
        .stApp {
            background-color: unset;
            color: unset;
        }
        h1, h3, p {
            color: unset !important;
        }
        </style>
        """, unsafe_allow_html=True)

# App Title
st.markdown("<h1 style='text-align: center;'>Depression Detection App</h1>", unsafe_allow_html=True)

# Navigation logic
if page == "Home":
    # Home Section
    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("<p style='font-style: italic; font-size: 18px;'>Welcome to the Depression Detection App!</p>", unsafe_allow_html=True)
    with col2:
        st.image("DP.jpeg", caption="", width=150)

    # Input Section: Add a text area for depression detection
    st.markdown("<h3 style='text-align: center;'>Enter your text for depression analysis:</h3>", unsafe_allow_html=True)

    # Ensure the text area has a label (even though it's hidden)
    user_input = st.text_area("Text for Analysis", placeholder="Enter text related to your thoughts or feelings.", label_visibility="collapsed")

    # Processing Section: Depression detection logic
    if st.button("Detect"):
        if user_input.strip():
            # Preprocess the input for the model
            inputs = tokenizer(user_input, return_tensors="pt", truncation=True, padding=True)
            outputs = model(**inputs)
            logits = outputs.logits
            prediction = torch.argmax(logits, dim=1).item()

            # Map prediction to labels
            if prediction == 1:
                result = "Depression detected. Please consider seeking help from a professional if needed."
            else:
                result = "No depression detected. Keep taking care of yourself!"

            # Display result
            st.success(result)
        else:
            st.error("Please enter some text for analysis.")

elif page == "Settings":
    # Settings Section: Allow user to change mode
    st.markdown("<h3 style='text-align: center;'>Settings</h3>", unsafe_allow_html=True)

    # Add red color styling for the radio button label
    st.markdown("<h4 style='color:red; text-align:center;'>Select Mode</h4>", unsafe_allow_html=True)

    # Custom CSS to change radio button labels to red
    st.markdown("""
    <style>
    .css-1y4pff9 {
        color: red !important;
    }
    </style>
    """, unsafe_allow_html=True)

    # Radio button for selecting mode
    mode = st.radio("", ["Dark Mode", "Light Mode", "System Mode"])

    # Set styles based on the selected mode
    set_styles(mode)
