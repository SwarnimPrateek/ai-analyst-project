import streamlit as st
import pandas as pd
import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_experimental.agents import create_pandas_dataframe_agent

import streamlit as st
import os
from langchain_google_genai import ChatGoogleGenerativeAI

# Use st.secrets if available (Cloud), otherwise fallback to os.getenv (Local)
api_key = st.secrets.get("GOOGLE_API_KEY") or os.getenv("GOOGLE_API_KEY")

if not api_key:
    st.error("API Key not found. Please set 'GOOGLE_API_KEY' in Streamlit Secrets.")
    st.stop()

# Initialize the model with the key
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash", 
    google_api_key=api_key, 
    temperature=0
)

# --- FOOLPROOF THEME CREATOR ---
# This forces Windows to create the folder and file in the exact right place
os.makedirs(".streamlit", exist_ok=True)
with open(".streamlit/config.toml", "w") as f:
    f.write("""
[theme]
primaryColor="#8b5cf6"
backgroundColor="#09090b"
secondaryBackgroundColor="#18181b"
textColor="#f8fafc"
font="sans serif"
""")

# Load environment variables
load_dotenv()

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="DataSight AI | Swarnim Prateek", page_icon="✨", layout="wide")

# --- SIDEBAR ---
with st.sidebar:
    st.markdown("## ⚙️ Configuration")
    uploaded_file = st.file_uploader("Drop CSV file here", type=["csv"])
    st.divider()
    st.caption("Built by **Swarnim Prateek**.")

# --- MAIN DASHBOARD ---
st.title("✨ AI-Powered Data Analyst")

if uploaded_file is None:
    st.info("👈 Please upload a CSV file in the sidebar to get started.")
else:
    df = pd.read_csv(uploaded_file)
    
    # Initialize the Gemini Model
    # --- INITIALIZE GEMINI ---
# This looks for the key in Streamlit Secrets FIRST, then checks the environment
try:
    api_key = st.secrets["GOOGLE_API_KEY"]
except:
    api_key = os.getenv("GOOGLE_API_KEY")

if not api_key:
    st.error("Missing API Key! Ensure it is set in Streamlit Cloud Secrets.")
    st.stop()

# Initialize the model with the explicitly retrieved key
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash", 
    temperature=0, 
    google_api_key=api_key
)

    col_data, col_chat = st.columns([1, 1], gap="large")
    
    with col_data:
        st.subheader("📊 Dataset Overview")
        st.dataframe(df.head(10), use_container_width=True)
        st.write(df.describe())
        
    with col_chat:
        st.subheader("🤖 Ask the Data")
        user_query = st.chat_input("Ask a question about your data...")
        
        if user_query:
            with st.chat_message("user"):
                st.write(user_query)
            
            with st.spinner("Analyzing..."):
                response = agent.invoke(user_query)
                
                with st.chat_message("assistant"):
                    st.write(response['output'])