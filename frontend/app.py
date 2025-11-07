# frontend/app.py
import streamlit as st
import requests
from dotenv import load_dotenv
import os

load_dotenv()

# Backend URL
API_BASE = os.getenv('API_BASE')

st.title("📘 Nano Notebook LLM")
st.write("Upload your PDF and ask questions!")

# PDF upload
uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])

if uploaded_file:
    with st.spinner("Uploading..."):
        files = {"file": uploaded_file.getbuffer()}
        res = requests.post(f"{API_BASE}/upload", files={"file": uploaded_file})
        if res.status_code == 200:
            st.success("✅ File uploaded and indexed successfully!")
        else:
            st.error("Upload failed. Check backend logs.")

# Ask question
question = st.text_input("Ask a question about your document")

if st.button("Ask"):
    if not question.strip():
        st.warning("Please enter a question.")
    else:
        with st.spinner("Querying the document..."):
            data = {"question": question}
            res = requests.post(f"{API_BASE}/query", data=data)
            if res.status_code == 200:
                st.success("Answer:")
                st.write(res.json()["response"])
            else:
                st.error("Error querying the document.")
