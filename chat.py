import chardet
import pandas as pd
import streamlit as st
from pandasai.llm.local_llm import LocalLLM
from pandasai import SmartDataframe

# Initialize the model
model = LocalLLM(
    api_base="https://231e-2a09-bac5-3b23-254b-00-3b7-48.ngrok-free.app",
    model="llama3"
)

st.title("Data analysis with PandasAI")

# File uploader
uploaded_file = st.file_uploader("Upload a CSV file", type=['csv'])

if uploaded_file is not None:
    # Detect encoding of the uploaded file
    uploaded_file.seek(0)  # Reset file pointer before reading
    raw_data = uploaded_file.read()
    result = chardet.detect(raw_data)
    encoding = result['encoding']
    print(f"Detected encoding: {encoding}")

    # Try reading the file with the detected encoding, fallback to ISO-8859-1 if needed
    try:
        uploaded_file.seek(0)  # Reset file pointer before reading again
        data = pd.read_csv(uploaded_file, encoding=encoding)
        st.success(f"File successfully read with {encoding} encoding.")
    except UnicodeDecodeError:
        print(f"Error reading file with encoding {encoding}, trying ISO-8859-1...")
        uploaded_file.seek(0)  # Reset file pointer
        data = pd.read_csv(uploaded_file, encoding='ISO-8859-1')
        st.success(f"File successfully read with ISO-8859-1 encoding.")

    # Show the first few rows of the data
    st.write(data.head(5))

    # Check for missing values in the data
    missing_values = data.isnull().sum()
    st.write(f"Missing values in the dataset: {missing_values}")

    # Optionally, drop rows with missing values or handle them
    # data = data.dropna()  # Uncomment this line to drop rows with missing values
    # st.write(data.head(5))

    # Initialize the SmartDataframe with the model
    df = SmartDataframe(data, config={"llm": model})

    # Text area to enter the prompt for the AI
    prompt = st.text_area("Enter your prompt:")

    # Generate response based on the user's prompt
    if st.button("Generate"):
        if prompt:
            with st.spinner("Generating response..."):
                st.write(df.chat(prompt))