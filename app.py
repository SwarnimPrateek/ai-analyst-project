import streamlit as st
import pandas as pd
import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_experimental.agents import create_pandas_dataframe_agent

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="DataSight AI | Swarnim Prateek", page_icon="✨", layout="wide")

# --- LOAD SECRETS ---
load_dotenv()
api_key = st.secrets.get("GOOGLE_API_KEY") or os.getenv("GOOGLE_API_KEY")

if not api_key:
    st.error("API Key not found. Please set 'GOOGLE_API_KEY' in Streamlit Secrets.")
    st.stop()

# --- INITIALIZE MODEL ---
llm = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash",  # <--- Updated model name
    google_api_key=api_key, 
    temperature=0
)

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
    
    # Create the agent here, after the file is uploaded
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
                try:
                    response = agent.invoke(user_query)
                    with st.chat_message("assistant"):
                        st.write(response['output'])
                except Exception as e:
                    st.error(f"An error occurred: {e}")