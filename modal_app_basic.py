"""
Production Modal implementation for the LangGraph Research Assistant.
"""
import os
import json
import modal
import uuid
import datetime
from typing import Dict, List, Any

# Create the Modal app
app = modal.App("research-assistant")

# Define image with dependencies
image = modal.Image.debian_slim().pip_install(
    "langchain>=0.0.267", 
    "langchain_core>=0.0.16",
    "langgraph>=0.0.15",
    "langchain_openai>=0.0.2", 
    "pinecone",  # Use the correct Pinecone package
    "python-dotenv>=1.0.0",
    "pydantic>=2.4.2",
    "fastapi>=0.103.1",
    "uvicorn>=0.23.2",
    "langchain-community>=0.0.10",
    "tavily-python>=0.1.8",
    "wikipedia>=1.4.0",
)

# GPU image for LLM operations
gpu_image = image.pip_install("torch", "transformers")

# Volume for persistent storage
volume = modal.Volume.from_name("research-assistant-vol", create_if_missing=True)

# API endpoint to verify the app is running
@app.function(image=image)
@modal.fastapi_endpoint()
def test_api():
    """Simple endpoint to test if the API is running."""
    return {"status": "ok", "message": "API is working"}

# Initialize the LLM
@app.function(image=gpu_image, gpu="T4", secrets=[modal.Secret.from_name("research-assistant-secrets")])
def setup_llm():
    """Initialize and return LLM configuration."""
    from langchain_openai import ChatOpenAI
    
    # Create LLM with API key from secrets
    model = os.environ.get("OPENAI_MODEL", "gpt-3.5-turbo")
    temperature = float(os.environ.get("OPENAI_TEMPERATURE", 0))
    max_tokens = int(os.environ.get("OPENAI_MAX_TOKENS", 1000))
    
    # Return configuration
    return {
        "model": model,
        "temperature": temperature,
        "max_tokens": max_tokens
    }

# Create analyst personas
@app.function(image=gpu_image, gpu="T4", secrets=[modal.Secret.from_name("research-assistant-secrets")])
def create_research_analysts(topic: str, num_analysts: int = 3):
    """Create analyst personas for a research topic."""
    from langchain_openai import ChatOpenAI
    from langchain_core.messages import SystemMessage, HumanMessage
    
    # Get LLM config and create LLM
    llm_config = setup_llm.remote()
    llm = ChatOpenAI(
        model=llm_config["model"],
        temperature=llm_config["temperature"],
        max_tokens=llm_config["max_tokens"]
    )
    
    # Instructions for analyst creation
    instructions = f"""
    You are tasked with creating a set of AI analyst personas. Follow these instructions carefully:

    1. First, review the research topic: {topic}
    2. Create exactly {num_analysts} total analysts.
    3. Each analyst should have a unique perspective and expertise that contributes valuable insights to the topic.
    4. Design analysts with diverse backgrounds, areas of expertise, and viewpoints to provide comprehensive coverage.
    5. Each analyst should focus on different aspects or implications of the research topic.
    
    For each analyst, provide:
    - Name
    - Affiliation 
    - Role
    - Description
    """
    
    # Generate analysts
    response = llm.invoke([
        SystemMessage(content=instructions),
        HumanMessage(content="Generate the set of analysts.")
    ])
    
    # Parse the response text to extract analysts
    analysts = []
    analyst_data = response.content.split("\n\n")
    
    for data in analyst_data:
        if "Name:" in data:
            lines = data.split("\n")
            analyst = {}
            for line in lines:
                if ":" in line:
                    key, value = line.split(":", 1)
                    key = key.strip().lower()
                    value = value.strip()
                    if key == "name":
                        analyst["name"] = value
                    elif key == "affiliation":
                        analyst["affiliation"] = value
                    elif key == "role":
                        analyst["role"] = value
                    elif key == "description":
                        analyst["description"] = value
            if len(analyst) >= 4:  # Ensure we have all required fields
                analysts.append(analyst)
    
    # Limit to the requested number
    return analysts[:num_analysts]

# Create a research job endpoint
@app.function(image=image, secrets=[modal.Secret.from_name("research-assistant-secrets")])
@modal.fastapi_endpoint(method="POST")
def create_research(request_data: Dict):
    """Start a new research job."""
    from pinecone import Pinecone
    
    # Extract request data
    topic = request_data.get("topic")
    num_analysts = request_data.get("num_analysts", 3)
    
    if not topic:
        return {"error": "Topic is required"}, 400
    
    # Generate job ID
    job_id = f"job_{uuid.uuid4().hex[:8]}"
    print(f"Creating job: {job_id}")
    
    # Initialize Pinecone with API key from secrets
    api_key = os.environ.get("PINECONE_API_KEY")
    if not api_key:
        return {"error": "Pinecone API key not configured"}, 500
    
    pc = Pinecone(api_key=api_key)
    index_name = os.environ.get("PINECONE_INDEX", "langgraph-jobs")
    
    # Connect to the index
    try:
        indexes = pc.list_indexes()
        index_names = [idx.name for idx in indexes]
        
        if index_name not in index_names:
            return {"error": f"Pinecone index {index_name} does not exist"}, 500
        
        # Connect to the index
        index = pc.Index(index_name)
        
        # Create job metadata
        metadata = {
            "topic": topic,
            "num_analysts": num_analysts,
            "status": "starting",
            "progress": 0.0,
            "current_step": "Initializing",
            "created_at": str(datetime.datetime.now())
        }
        
        # Create a vector
        vector = [0.0] * 1535 + [0.1]  # Non-zero vector
        
        # Insert into Pinecone
        index.upsert(
            vectors=[{
                "id": job_id,
                "values": vector,
                "metadata": metadata
            }],
            namespace="jobs"
        )
        print(f"Job saved to Pinecone: {job_id}")
        
        # Start the research job asynchronously
        process_research_job.spawn(job_id, topic, num_analysts)
        
        return {"job_id": job_id, "status": "starting"}
    
    except Exception as e:
        print(f"Error creating job: {str(e)}")
        return {"error": f"Error starting job: {str(e)}"}, 500

# Get research status endpoint
@app.function(image=image, secrets=[modal.Secret.from_name("research-assistant-secrets")])
@modal.fastapi_endpoint(method="GET")
def get_research_status(job_id: str):
    """Get the status of a research job."""
    from pinecone import Pinecone
    
    print(f"Checking status for job_id: {job_id}")
    
    # Initialize Pinecone
    api_key = os.environ.get("PINECONE_API_KEY")
    if not api_key:
        return {"error": "Pinecone API key not configured"}, 500
    
    pc = Pinecone(api_key=api_key)
    index_name = os.environ.get("PINECONE_INDEX", "langgraph-jobs")
    
    # Connect to the index
    try:
        index = pc.Index(index_name)
        
        # Fetch job data
        result = index.fetch(
            ids=[job_id],
            namespace="jobs"
        )
        
        # Check if job exists
        if not result or not hasattr(result, "vectors") or job_id not in result.vectors:
            print(f"Job not found in Pinecone: {job_id}")
            return {"error": "Job not found"}, 404
        
        # Extract job data
        vector_data = result.vectors[job_id]
        metadata = vector_data.metadata if hasattr(vector_data, "metadata") else {}
        
        # Prepare response
        response = {
            "job_id": job_id,
            "status": metadata.get("status", "unknown"),
            "progress": float(metadata.get("progress", 0.0)),
            "current_step": metadata.get("current_step", ""),
        }
        
        # Include analysts if available
        if "analysts" in metadata:
            response["analysts"] = metadata["analysts"]
        
        # Include final report if job is complete
        if metadata.get("status") == "completed" and metadata.get("has_report"):
            # Fetch report data
            report_result = index.fetch(
                ids=[f"report_{job_id}"],
                namespace="reports"
            )
            
            if hasattr(report_result, "vectors") and f"report_{job_id}" in report_result.vectors:
                report_data = report_result.vectors[f"report_{job_id}"]
                if hasattr(report_data, "metadata") and "report" in report_data.metadata:
                    response["final_report"] = report_data.metadata["report"]
        
        # Include error if any
        if metadata.get("status") == "error" and "error" in metadata:
            response["error"] = metadata["error"]
        
        return response
    
    except Exception as e:
        print(f"Error fetching job status: {str(e)}")
        return {"error": f"Error fetching job status: {str(e)}"}, 500

# Process the research job asynchronously
@app.function(image=image, timeout=3600, secrets=[modal.Secret.from_name("research-assistant-secrets")])
def process_research_job(job_id: str, topic: str, num_analysts: int):
    """Process the research job in the background."""
    import traceback
    from pinecone import Pinecone
    
    print(f"Starting research job processing: {job_id}")
    import time
    time.sleep(3) 
    
    # Initialize Pinecone
    api_key = os.environ.get("PINECONE_API_KEY")
    pc = Pinecone(api_key=api_key)
    index_name = os.environ.get("PINECONE_INDEX", "langgraph-jobs")
    index = pc.Index(index_name)
    
    # Helper function to update job status
    def update_job(updates):
        try:
            # Get current metadata
            result = index.fetch(
                ids=[job_id],
                namespace="jobs"
            )
            
            if not result or not hasattr(result, "vectors") or job_id not in result.vectors:
                print(f"Failed to fetch job for updating: {job_id}")
                return False
            
            # Get existing metadata
            vector_data = result.vectors[job_id]
            metadata = vector_data.metadata if hasattr(vector_data, "metadata") else {}
            
            # Update metadata
            metadata.update(updates)
            
            # Update in Pinecone
            index.upsert(
                vectors=[{
                    "id": job_id,
                    "values": [0.0] * 1535 + [0.1],  # Non-zero vector
                    "metadata": metadata
                }],
                namespace="jobs"
            )
            print(f"Updated job {job_id} with: {updates}")
            return True
        except Exception as e:
            print(f"Error updating job: {e}")
            return False
    
    # Helper function to save report
    def save_report(report_content):
        try:
            index.upsert(
                vectors=[{
                    "id": f"report_{job_id}",
                    "values": [0.0] * 1535 + [0.1],
                    "metadata": {"report": report_content}
                }],
                namespace="reports"
            )
            print(f"Saved report for job {job_id}")
            return True
        except Exception as e:
            print(f"Error saving report: {e}")
            return False
    
    try:
        # Update status to running
        update_job({
            "status": "running",
            "progress": 0.1,
            "current_step": "Creating analysts"
        })
        
        # Step 1: Create analysts
        analysts = create_research_analysts.remote(topic, num_analysts)
        
        # Save analysts
        update_job({
            "progress": 0.2,
            "current_step": "Analysts created",
            "analysts": analysts
        })
        
        # Step 2: Generate a report
        # In a full implementation, this would use LangGraph to orchestrate the research process
        update_job({
            "progress": 0.5,
            "current_step": "Conducting research and generating report"
        })
        
        report = generate_report.remote(topic, analysts)
        
        # Save the report
        save_report(report)
        
        # Update job status to complete
        update_job({
            "progress": 1.0,
            "current_step": "Complete",
            "status": "completed",
            "has_report": True,
            "completed_at": str(datetime.datetime.now())
        })
        
    except Exception as e:
        error_details = traceback.format_exc()
        print(f"Error in research job: {error_details}")
        
        # Update job status on error
        update_job({
            "status": "error",
            "error": str(e)
        })

# Generate a report
@app.function(image=gpu_image, gpu="T4", secrets=[modal.Secret.from_name("research-assistant-secrets")])
def generate_report(topic: str, analysts: List[Dict]):
    """Generate a research report based on the topic and analysts."""
    from langchain_openai import ChatOpenAI
    
    # Get LLM config
    llm_config = setup_llm.remote()
    
    # Create the LLM
    llm = ChatOpenAI(
        model=llm_config["model"],
        temperature=llm_config["temperature"],
        max_tokens=llm_config["max_tokens"]
    )
    
    # Format analyst information
    analyst_info = ""
    for i, analyst in enumerate(analysts, 1):
        analyst_info += f"\nAnalyst {i}:\n"
        analyst_info += f"Name: {analyst['name']}\n"
        analyst_info += f"Role: {analyst['role']}\n"
        analyst_info += f"Affiliation: {analyst['affiliation']}\n"
        analyst_info += f"Description: {analyst['description']}\n\n"
    
    # Create report prompt
    prompt = f"""
    Generate a comprehensive research report on the topic: {topic}
    
    The report should incorporate perspectives from the following analysts:
    {analyst_info}
    
    The report should include:
    1. A title
    2. An introduction
    3. Main sections covering different aspects of the topic
    4. A conclusion
    5. Sources section
    
    Format the report in Markdown.
    """
    
    # Generate the report
    response = llm.invoke(prompt)
    
    return response.content

if __name__ == "__main__":
    # For local testing
    print("Testing Modal functions locally...")
    
    with app.run():
        # Test API connection
        print("API status:", test_api.remote())
        
        # Test LLM setup
        print("LLM config:", setup_llm.remote())