import streamlit as st
import requests
import time
import json
import os
import markdown
import weasyprint
from io import BytesIO
from dotenv import load_dotenv

# Load environment variables (for local development)
if not os.environ.get("MODAL_ENVIRONMENT"):
    load_dotenv()

# Configure page
st.set_page_config(
    page_title="LangGraph Research Assistant",
    page_icon="📊",
    layout="wide"
)

# Modal API URLs - production deployment
BASE_API_URL = "https://silverside-ai--research-assistant"
API_ENDPOINTS = {
    "test": f"{BASE_API_URL}-test-api.modal.run",
    "create": f"{BASE_API_URL}-create-research.modal.run",
    "status": f"{BASE_API_URL}-get-research-status.modal.run"
}

# Initialize session state
if 'job_id' not in st.session_state:
    st.session_state.job_id = None
if 'poll_status' not in st.session_state:
    st.session_state.poll_status = False
if 'job_complete' not in st.session_state:
    st.session_state.job_complete = False
if 'analysts' not in st.session_state:
    st.session_state.analysts = None
if 'final_report' not in st.session_state:
    st.session_state.final_report = None
if 'progress' not in st.session_state:
    st.session_state.progress = 0
if 'current_step' not in st.session_state:
    st.session_state.current_step = ""
if 'debug_log' not in st.session_state:
    st.session_state.debug_log = []

# Logging function
def log_to_debug(message):
    timestamp = time.strftime("%H:%M:%S")
    st.session_state.debug_log.append(f"[{timestamp}] {message}")

# Helper function to display analysts
def display_analysts(analysts):
    if not analysts:
        return
    
    for i, analyst in enumerate(analysts):
        with st.container():
            st.subheader(analyst["name"])
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"**Role:** {analyst['role']}")
                st.write(f"**Affiliation:** {analyst['affiliation']}")
            with col2:
                st.write(f"**Description:** {analyst['description']}")
            st.divider()
            log_to_debug(f"Displayed analyst {i+1}: {analyst['name']}")

# Poll job status
def poll_job_status():
    if not st.session_state.job_id or st.session_state.job_complete:
        return
    
    try:
        response = requests.get(f"{API_ENDPOINTS['status']}?job_id={st.session_state.job_id}")
        
        if response.status_code == 200:
            data = response.json()
            
            # Check if response is a list (error case)
            if isinstance(data, list):
                log_to_debug(f"Received error response: {data}")
                if len(data) > 0 and isinstance(data[0], dict) and "error" in data[0]:
                    st.error(f"Error: {data[0]['error']}")
                return False
                
            log_to_debug(f"Poll status: {data.get('status', 'unknown')}, progress: {data.get('progress', 0)}")
            
            # Update state
            st.session_state.progress = data.get("progress", 0)
            st.session_state.current_step = data.get("current_step", "Processing")
            
            # Update analysts if available
            if data.get("analysts") and not st.session_state.analysts:
                st.session_state.analysts = data["analysts"]
                log_to_debug(f"Received {len(data['analysts'])} analysts")
                
            # Check if job completed
            if data.get("status") == "completed" and data.get("final_report"):
                st.session_state.final_report = data["final_report"]
                st.session_state.job_complete = True
                log_to_debug("Research completed")
                return True
                
            # Check for errors
            if data.get("status") == "error":
                st.error(f"Error in research process: {data.get('error', 'Unknown error')}")
                st.session_state.poll_status = False
                log_to_debug(f"Research error: {data.get('error', 'Unknown error')}")
                return False
        else:
            log_to_debug(f"Error polling job status: {response.status_code} - {response.text}")
            return False
            
    except Exception as e:
        st.error(f"Error polling job status: {str(e)}")
        log_to_debug(f"Polling error: {str(e)}")
        return False
    
    return False  # Not complete yet

# Main app interface
st.title("🐟 Silverside Research Assistant")
st.write("Generate comprehensive research reports with AI expert analysts!")

# Sidebar for configuration and logs
with st.sidebar:
    st.header("Configuration")
    
    # API connection test
    if st.button("Test API Connection"):
        try:
            response = requests.get(API_ENDPOINTS["test"])
            if response.status_code == 200:
                st.success("API connection successful!")
                log_to_debug(f"API connection test successful")
            else:
                st.error(f"API connection failed: {response.status_code}")
                log_to_debug(f"API connection test failed - {response.status_code}")
        except Exception as e:
            st.error(f"API connection error: {str(e)}")
            log_to_debug(f"API connection test error - {str(e)}")
    
    # Show API URL for debugging
    with st.expander("API Endpoints"):
        st.code(f"Test API: {API_ENDPOINTS['test']}")
        st.code(f"Create Research: {API_ENDPOINTS['create']}")
        st.code(f"Get Status: {API_ENDPOINTS['status']}")
    
    # Current status
    if st.session_state.job_id:
        st.info(f"Current job ID: {st.session_state.job_id}")
    
    # Debug logs
    with st.expander("Debug Log"):
        for log_entry in st.session_state.debug_log:
            st.text(log_entry)

# Research topic input
st.header("Research Topic")
topic = st.text_input("Enter the research topic you want to analyze:", 
                     placeholder="Example: The impact of AI on healthcare")

# Number of analysts
num_analysts = st.slider("Number of analyst personas:", min_value=2, max_value=5, value=3)

# Run button
if st.button("Generate Research Report", type="primary", disabled=(not topic)):
    if not topic:
        st.warning("Please enter a research topic.")
    else:
        with st.spinner(f"Starting research on '{topic}'..."):
            try:
                # Reset state for new research
                st.session_state.analysts = None
                st.session_state.final_report = None
                st.session_state.job_complete = False
                
                # Call API to start research
                log_to_debug(f"Starting research on topic: {topic} with {num_analysts} analysts")
                response = requests.post(
                    API_ENDPOINTS["create"],
                    json={"topic": topic, "num_analysts": num_analysts}
                )
                
                if response.status_code == 200:
                    data = response.json()
                    st.session_state.job_id = data["job_id"]
                    st.session_state.poll_status = True
                    log_to_debug(f"Research job created with ID: {data['job_id']}")
                    st.rerun()  # Refresh to show progress bar
                else:
                    st.error(f"Error starting research: {response.text}")
                    log_to_debug(f"API error: {response.text}")
            except Exception as e:
                st.error(f"Error calling API: {str(e)}")
                log_to_debug(f"Exception: {str(e)}")

# Progress tracking
if st.session_state.job_id and not st.session_state.job_complete:
    # Display progress bar
    st.progress(st.session_state.progress)
    st.info(f"Status: {st.session_state.current_step}")
    
    # Set up polling - only do this when displaying the page
    if st.session_state.poll_status:
        complete = poll_job_status()
        if complete:
            st.rerun()  # Refresh to show final report
        else:
            # Add auto-refresh for polling (every 3 seconds)
            time.sleep(3)
            st.rerun()

# Display analysts if available
if st.session_state.analysts:
    with st.expander("Analyst Personas", expanded=not st.session_state.final_report):
        display_analysts(st.session_state.analysts)

# Display final report if available
if st.session_state.final_report:
    st.header("Research Report")
    st.markdown(st.session_state.final_report)
    log_to_debug("Displayed final report")
    
    # Download buttons
    col1, col2 = st.columns(2)
    
    with col1:
        # Markdown download
        st.download_button(
            label="Download as Markdown",
            data=st.session_state.final_report,
            file_name=f"research_report_{topic.replace(' ', '_')}.md",
            mime="text/markdown"
        )
    
    with col2:
        try:
            # Convert markdown to HTML
            html = markdown.markdown(st.session_state.final_report)
            
            # Convert HTML to PDF
            pdf_bytes = weasyprint.HTML(string=html).write_pdf()
            
            # Download button
            st.download_button(
                label="Download as PDF",
                data=pdf_bytes,
                file_name=f"research_report_{topic.replace(' ', '_')}.pdf",
                mime="application/pdf"
            )
        except Exception as e:
            st.warning(f"PDF generation failed: {str(e)}")
            log_to_debug(f"PDF generation error: {str(e)}")