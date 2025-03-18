import os
import streamlit as st
from dotenv import load_dotenv
from gemini_agent import process_user_query

# Load environment variables
load_dotenv()

# Set up the Streamlit app
st.set_page_config(
    page_title="Canvas AI Assistant",
    page_icon="🎓",
    layout="wide"
)

# App title and description
st.title("Canvas AI Assistant")
st.markdown("""
This AI assistant helps you interact with your Canvas LMS using natural language.
Ask questions about your classes, assignments, grades, and more!
""")

# Sidebar with information
with st.sidebar:
    st.header("About")
    st.info("""
    This assistant uses Google's Gemini AI to understand your questions and dynamically
    select the appropriate tools to fetch information from Canvas LMS.
    
    Powered by:
    - Gemini Pro
    - LangChain
    - LangGraph
    - Canvas API
    """)
    
    st.header("Example Questions")
    st.markdown("""
    - What are my current classes?
    - Show me my upcoming assignments for Biology
    - What's my current grade in Chemistry?
    - When is my next test in Math?
    - Show me the syllabus for History
    """)

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Ask about your Canvas classes..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Display assistant response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = process_user_query(prompt)
            st.markdown(response)
    
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})
