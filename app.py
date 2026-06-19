import streamlit as st
import pandas as pd
import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_experimental.agents import create_pandas_dataframe_agent

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
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)
    
    # Create the AI Agent
    agent = create_pandas_dataframe_agent(
        llm, 
        df, 
        verbose=True, 
        allow_dangerous_code=True
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