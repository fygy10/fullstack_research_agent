# # Load environment variables first, before any other imports
# import os
# from dotenv import load_dotenv
# load_dotenv()  # This loads the .env file at the module level

# from fastapi import FastAPI, HTTPException, BackgroundTasks
# from pydantic import BaseModel
# import uuid
# import datetime
# from typing import Dict, Any, List, Optional
# import functools

# # Define request/response models
# class ResearchRequest(BaseModel):
#     topic: str
#     num_analysts: int = 3

# class ResearchResponse(BaseModel):
#     job_id: str
#     status: str

# class ResearchStatus(BaseModel):
#     job_id: str
#     status: str
#     progress: float
#     current_step: str
#     analysts: Optional[List[dict]] = None
#     final_report: Optional[str] = None

# # Pinecone client for job tracking
# class PineconeJobTracker:
#     def __init__(self, api_key, index_name="langgraph-jobs", namespace="jobs"):
#         from pinecone import Pinecone, ServerlessSpec
#         self.pc = Pinecone(api_key=api_key)
#         self.index_name = index_name
#         self.namespace = namespace
        
#         # Initialize index if it doesn't exist
#         try:
#             indexes = self.pc.list_indexes()
#             index_exists = False
#             index_names = [idx.name for idx in indexes]
            
#             if self.index_name not in index_names:
#                 print(f"Creating Pinecone index: {self.index_name}")
#                 self.pc.create_index(
#                     name=self.index_name,
#                     dimension=1536,
#                     metric="cosine",
#                     spec=ServerlessSpec(
#                         cloud="aws",
#                         region="us-east-1"
#                     )
#                 )
                
#             # Connect to the index
#             self.index = self.pc.Index(self.index_name)
#             print(f"Connected to Pinecone index: {self.index_name}")
#         except Exception as e:
#             raise ConnectionError(f"Failed to connect to Pinecone index: {e}")
    
#     def create_job(self, job_id, topic, num_analysts):
#         # Create a vector with at least one non-zero value
#         vector = [0.0] * 1535 + [0.1]  # Add a small non-zero value
        
#         # Create metadata
#         metadata = {
#             "topic": topic,
#             "num_analysts": num_analysts,
#             "status": "starting",
#             "progress": 0.0,
#             "current_step": "Initializing",
#             "created_at": str(datetime.datetime.now())
#         }
        
#         # Insert into Pinecone
#         self.index.upsert(
#             vectors=[{
#                 "id": job_id,
#                 "values": vector,
#                 "metadata": metadata
#             }],
#             namespace=self.namespace
#         )
        
#         return job_id
    
#     def update_job(self, job_id, updates):
#         # Get current data
#         try:
#             result = self.index.fetch(
#                 ids=[job_id],
#                 namespace=self.namespace
#             )
            
#             # New API returns a FetchResponse object
#             # Check if result has vectors for the job_id
#             if not result or not hasattr(result, "vectors") or job_id not in result.vectors:
#                 return False
            
#             # Get existing metadata - access as property on result.vectors[job_id]
#             vector_data = result.vectors[job_id]
#             metadata = vector_data.metadata if hasattr(vector_data, "metadata") else {}
            
#             # Update metadata
#             metadata.update(updates)
            
#             # Update in Pinecone
#             self.index.upsert(
#                 vectors=[{
#                     "id": job_id,
#                     "values": [0.0] * 1535 + [0.1],  # Non-zero vector
#                     "metadata": metadata
#                 }],
#                 namespace=self.namespace
#             )
            
#             return True
#         except Exception as e:
#             print(f"Error updating job: {e}")
#             return False
    
#     def get_job(self, job_id):
#         try:
#             result = self.index.fetch(
#                 ids=[job_id],
#                 namespace=self.namespace
#             )
            
#             # Check if the job_id exists in the result
#             if not result or not hasattr(result, "vectors") or job_id not in result.vectors:
#                 return None
            
#             # Get the metadata from the vector
#             vector_data = result.vectors[job_id]
            
#             # Access metadata as a property
#             if hasattr(vector_data, "metadata"):
#                 return vector_data.metadata
#             return {}
            
#         except Exception as e:
#             print(f"Error getting job: {e}")
#             return None
    
#     def save_report(self, job_id, report):
#         # For longer content like reports, we need to save in a separate namespace
#         try:
#             self.index.upsert(
#                 vectors=[{
#                     "id": f"report_{job_id}",
#                     "values": [0.0] * 1535 + [0.1],  # Non-zero vector
#                     "metadata": {"report": report}
#                 }],
#                 namespace="reports"
#             )
            
#             self.update_job(job_id, {"has_report": True})
#             return True
#         except Exception as e:
#             print(f"Error saving report: {e}")
#             return False
    
# # In api_main.py, find the PineconeJobTracker class and replace the get_report method:

#     def get_report(self, job_id):
#         try:
#             print(f"Fetching report for job {job_id}")
#             result = self.index.fetch(
#                 ids=[f"report_{job_id}"],
#                 namespace="reports"
#             )
            
#             # Check if report exists
#             if not hasattr(result, "vectors") or f"report_{job_id}" not in result.vectors:
#                 print(f"No report found with ID report_{job_id}")
#                 return None
            
#             # Extract vector data
#             vector_data = result.vectors[f"report_{job_id}"]
            
#             # Get metadata and extract report
#             if hasattr(vector_data, "metadata"):
#                 metadata = vector_data.metadata
                
#                 # Try different ways to access the report content
#                 if hasattr(metadata, "report"):
#                     return metadata.report
#                 elif isinstance(metadata, dict) and "report" in metadata:
#                     return metadata["report"]
            
#             print(f"Report found but couldn't extract content")
#             return None
            
#         except Exception as e:
#             print(f"Error getting report: {e}")
#             return None


#     def get_analysts(self, job_id):
#         """Get analysts for a job."""
#         try:
#             # Try to get analysts from job metadata first
#             job_data = self.get_job(job_id)
#             if job_data and "analysts" in job_data:
#                 return job_data["analysts"]
                
#             # If not found, try dedicated analysts vector
#             result = self.index.fetch(
#                 ids=[f"analysts_{job_id}"],
#                 namespace="job_analysts"
#             )
            
#             if hasattr(result, "vectors") and f"analysts_{job_id}" in result.vectors:
#                 vector_data = result.vectors[f"analysts_{job_id}"]
#                 if hasattr(vector_data, "metadata") and "analysts" in vector_data.metadata:
#                     # Deserialize JSON
#                     analysts_json = vector_data.metadata["analysts"]
#                     return json.loads(analysts_json)
            
#             return None
#         except Exception as e:
#             print(f"Error retrieving analysts: {e}")
#             return None

# # Initialize API with required components
# def create_api():
#     # Import your research components
#     from set_env import setup_environment
#     from analysts import create_analysts
#     from build_graph import build_graph
#     import converse
#     import finalize
#     from langchain_community.tools.tavily_search import TavilySearchResults
    
#     # Initialize API
#     fastapi_app = FastAPI(title="Research Assistant API")
    
#     # Initialize Pinecone job tracker
#     job_tracker = PineconeJobTracker(
#         api_key=os.environ.get("PINECONE_API_KEY"),
#         index_name="langgraph-jobs"
#     )
    
#     # Set up LLM and tools
#     llm = setup_environment()
#     converse.llm = llm
#     finalize.llm = llm
#     converse.tavily_search = TavilySearchResults(max_results=3)
    
#     # Create the graph
#     create_analysts_with_llm = lambda state: create_analysts(state, llm)
#     graph = build_graph(create_analysts_with_llm)
    
#     # API endpoints
#     @fastapi_app.post("/api/research", response_model=ResearchResponse)
#     async def create_research(request: ResearchRequest, background_tasks: BackgroundTasks):
#         # Generate job ID
#         job_id = f"job_{uuid.uuid4().hex[:8]}"
        
#         # Create job in Pinecone
#         job_tracker.create_job(job_id, request.topic, request.num_analysts)
        
#         # Start research in background
#         background_tasks.add_task(
#             run_research_job, 
#             job_id, 
#             request.topic, 
#             request.num_analysts, 
#             graph,
#             job_tracker
#         )
        
#         return {"job_id": job_id, "status": "starting"}
    
# # In api_main.py, replace the get_research_status function with this:

# # Add these functions to your api_main.py file

#     # Add a test endpoint for API connectivity checks
#     @fastapi_app.get("/api/research/test")
#     async def test_api():
#         """Simple endpoint to test if the API is running"""
#         return {"status": "ok", "message": "API is working"}

#     # Enhance error handling in your existing endpoint
#     @fastapi_app.get("/api/research/{job_id}", response_model=ResearchStatus)
#     async def get_research_status(job_id: str):
#         # Get job from Pinecone
#         job_data = job_tracker.get_job(job_id)
        
#         if not job_data:
#             raise HTTPException(status_code=404, detail="Job not found")
        
#         # Get analysts (either from job data or dedicated vector)
#         analysts = job_data.get("analysts")
#         if not analysts:
#             analysts = job_tracker.get_analysts(job_id)
            
#         # Set up response
#         response = {
#             "job_id": job_id,
#             "status": job_data.get("status", "unknown"),
#             "progress": float(job_data.get("progress", 0.0)),
#             "current_step": job_data.get("current_step", ""),
#             "analysts": analysts
#         }
        
#         # Get final report if job is complete
#         if job_data.get("status") == "completed":
#             report = job_tracker.get_report(job_id)
#             if report:
#                 response["final_report"] = report
#             elif "final_report" in job_data:
#                 response["final_report"] = job_data["final_report"]
#             elif "report" in job_data:
#                 response["final_report"] = job_data["report"]
        
#         # Include error message if any
#         if job_data.get("status") == "error" and "error" in job_data:
#             response["error"] = job_data["error"]
        
#         return response
    
#     # Function to run research
#     def run_research_job(job_id, topic, num_analysts, graph, job_tracker):
#         try:
#             # Update status
#             job_tracker.update_job(job_id, {"status": "running"})
            
#             # Initial state
#             initial_state = {
#                 'topic': topic, 
#                 'number_analysts': num_analysts,
#                 'sections': []
#             }
            
#             # Thread ID for graph
#             thread_id = f"research_{topic.replace(' ', '_')}_{uuid.uuid4().hex[:8]}"
#             thread = {'configurable': {'thread_id': thread_id}}
            
#             # Track progress through graph execution
#             for event in graph.stream(initial_state, thread, stream_mode="updates"):
#                 node_name = next(iter(event.keys()))
                
#                 # Update progress based on node
#                 if node_name == "create_analysts":
#                     if 'analysts' in event[node_name]:
#                         # Convert analysts to a dictionary format that can be serialized
#                         analysts_data = [
#                             {k: v for k, v in (analyst.model_dump() if hasattr(analyst, 'model_dump') else analyst.dict()).items()} 
#                             for analyst in event[node_name]['analysts']
#                         ]
                        
#                         # Print for debugging
#                         print(f"Saving {len(analysts_data)} analysts to job {job_id}")
                        
#                         # First save analysts to job state
#                         job_tracker.update_job(job_id, {
#                             "analysts": analysts_data,
#                             "progress": 0.2,
#                             "current_step": "Conducting interviews",
#                             "analyst_count": len(analysts_data)
#                         })
                        
#                         # For extra reliability, save analysts in their own vector
#                         try:
#                             job_tracker.index.upsert(
#                                 vectors=[{
#                                     "id": f"analysts_{job_id}",
#                                     "values": [0.0] * 1535 + [1.0],  # Non-zero vector
#                                     "metadata": {"analysts": json.dumps(analysts_data)}
#                                 }],
#                                 namespace="job_analysts"
#                             )
#                             print(f"Saved analysts in separate vector")
#                         except Exception as e:
#                             print(f"Error saving analysts in separate vector: {e}")
                
#                 elif node_name == "write_report":
#                     job_tracker.update_job(job_id, {
#                         "progress": 0.6,
#                         "current_step": "Writing report"
#                     })
                
#                 elif node_name == "write_introduction":
#                     job_tracker.update_job(job_id, {
#                         "progress": 0.7,
#                         "current_step": "Writing introduction"
#                     })
                
#                 elif node_name == "write_conclusion":
#                     job_tracker.update_job(job_id, {
#                         "progress": 0.8,
#                         "current_step": "Writing conclusion"
#                     })
                
#                 elif node_name == "finalize_report":
#                     if 'final_report' in event[node_name]:
#                         # Save the report to Pinecone
#                         job_tracker.save_report(job_id, event[node_name]['final_report'])
                        
#                         # Update job status
#                         job_tracker.update_job(job_id, {
#                             "progress": 1.0,
#                             "current_step": "Complete",
#                             "status": "completed",
#                             "completed_at": str(datetime.datetime.now())
#                         })
            
#             # Ensure completion
#             job_status = job_tracker.get_job(job_id)
#             if job_status and float(job_status.get("progress", 0)) < 1.0:
#                 job_tracker.update_job(job_id, {
#                     "progress": 1.0,
#                     "current_step": "Complete",
#                     "status": "completed",
#                     "completed_at": str(datetime.datetime.now())
#                 })
                
#         except Exception as e:
#             import traceback
#             error_details = traceback.format_exc()
#             print(f"Error in research job: {error_details}")
#             # Update job status on error
#             job_tracker.update_job(job_id, {
#                 "status": "error",
#                 "error": str(e)
#             })
    
#     # Return the FastAPI app
#     return fastapi_app

# # Create app instance
# app = create_api()

# if __name__ == "__main__":
#     import uvicorn
    
#     # Debug environment variables
#     print(f"PINECONE_API_KEY exists: {bool(os.environ.get('PINECONE_API_KEY'))}")
#     print(f"OPENAI_API_KEY exists: {bool(os.environ.get('OPENAI_API_KEY'))}")
    
#     # Set up environment variables if not already set
#     os.environ.setdefault("STORAGE_TYPE", "pinecone")
#     os.environ.setdefault("PINECONE_INDEX", "langgraph_checkpoints")
    
#     # Run the app
#     uvicorn.run(app, host="0.0.0.0", port=8000)


# Load environment variables first, before any other imports
import os
from dotenv import load_dotenv
load_dotenv()  # This loads the .env file at the module level

from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
import uuid
import datetime
from typing import Dict, Any, List, Optional
import functools
import json

# Define request/response models
class ResearchRequest(BaseModel):
    topic: str
    num_analysts: int = 3

class ResearchResponse(BaseModel):
    job_id: str
    status: str

class ResearchStatus(BaseModel):
    job_id: str
    status: str
    progress: float
    current_step: str
    analysts: Optional[List[dict]] = None
    final_report: Optional[str] = None

# Pinecone client for job tracking
class PineconeJobTracker:
    def __init__(self, api_key, index_name="langgraph-jobs", namespace="jobs"):
        from pinecone import Pinecone, ServerlessSpec
        self.pc = Pinecone(api_key=api_key)
        self.index_name = index_name
        self.namespace = namespace
        
        # Initialize index if it doesn't exist
        try:
            indexes = self.pc.list_indexes()
            index_exists = False
            index_names = [idx.name for idx in indexes]
            
            if self.index_name not in index_names:
                print(f"Creating Pinecone index: {self.index_name}")
                self.pc.create_index(
                    name=self.index_name,
                    dimension=1536,
                    metric="cosine",
                    spec=ServerlessSpec(
                        cloud="aws",
                        region="us-east-1"
                    )
                )
                
            # Connect to the index
            self.index = self.pc.Index(self.index_name)
            print(f"Connected to Pinecone index: {self.index_name}")
        except Exception as e:
            raise ConnectionError(f"Failed to connect to Pinecone index: {e}")
    
    def create_job(self, job_id, topic, num_analysts):
        # Create a vector with at least one non-zero value
        vector = [0.0] * 1535 + [0.1]  # Add a small non-zero value
        
        # Create metadata
        metadata = {
            "topic": topic,
            "num_analysts": num_analysts,
            "status": "starting",
            "progress": 0.0,
            "current_step": "Initializing",
            "created_at": str(datetime.datetime.now())
        }
        
        # Insert into Pinecone
        self.index.upsert(
            vectors=[{
                "id": job_id,
                "values": vector,
                "metadata": metadata
            }],
            namespace=self.namespace
        )
        
        return job_id
    
    def update_job(self, job_id, updates):
        # Get current data
        try:
            result = self.index.fetch(
                ids=[job_id],
                namespace=self.namespace
            )
            
            # New API returns a FetchResponse object
            # Check if result has vectors for the job_id
            if not result or not hasattr(result, "vectors") or job_id not in result.vectors:
                return False
            
            # Get existing metadata - access as property on result.vectors[job_id]
            vector_data = result.vectors[job_id]
            metadata = vector_data.metadata if hasattr(vector_data, "metadata") else {}
            
            # Update metadata
            metadata.update(updates)
            
            # Update in Pinecone
            self.index.upsert(
                vectors=[{
                    "id": job_id,
                    "values": [0.0] * 1535 + [0.1],  # Non-zero vector
                    "metadata": metadata
                }],
                namespace=self.namespace
            )
            
            return True
        except Exception as e:
            print(f"Error updating job: {e}")
            return False
    
    def get_job(self, job_id):
        try:
            result = self.index.fetch(
                ids=[job_id],
                namespace=self.namespace
            )
            
            # Check if the job_id exists in the result
            if not result or not hasattr(result, "vectors") or job_id not in result.vectors:
                return None
            
            # Get the metadata from the vector
            vector_data = result.vectors[job_id]
            
            # Access metadata as a property
            if hasattr(vector_data, "metadata"):
                return vector_data.metadata
            return {}
            
        except Exception as e:
            print(f"Error getting job: {e}")
            return None
    
    def save_report(self, job_id, report):
        # For longer content like reports, we need to save in a separate namespace
        try:
            self.index.upsert(
                vectors=[{
                    "id": f"report_{job_id}",
                    "values": [0.0] * 1535 + [0.1],  # Non-zero vector
                    "metadata": {"report": report}
                }],
                namespace="reports"
            )
            
            self.update_job(job_id, {"has_report": True})
            return True
        except Exception as e:
            print(f"Error saving report: {e}")
            return False
    
    def get_report(self, job_id):
        try:
            print(f"Fetching report for job {job_id}")
            result = self.index.fetch(
                ids=[f"report_{job_id}"],
                namespace="reports"
            )
            
            # Check if report exists
            if not hasattr(result, "vectors") or f"report_{job_id}" not in result.vectors:
                print(f"No report found with ID report_{job_id}")
                return None
            
            # Extract vector data
            vector_data = result.vectors[f"report_{job_id}"]
            
            # Get metadata and extract report
            if hasattr(vector_data, "metadata"):
                metadata = vector_data.metadata
                
                # Try different ways to access the report content
                if hasattr(metadata, "report"):
                    return metadata.report
                elif isinstance(metadata, dict) and "report" in metadata:
                    return metadata["report"]
            
            print(f"Report found but couldn't extract content")
            return None
            
        except Exception as e:
            print(f"Error getting report: {e}")
            return None

    def get_analysts(self, job_id):
        """Get analysts for a job."""
        try:
            # Try to get analysts from job metadata first
            job_data = self.get_job(job_id)
            if job_data and "analysts" in job_data:
                return job_data["analysts"]
                
            # If not found, try dedicated analysts vector
            result = self.index.fetch(
                ids=[f"analysts_{job_id}"],
                namespace="job_analysts"
            )
            
            if hasattr(result, "vectors") and f"analysts_{job_id}" in result.vectors:
                vector_data = result.vectors[f"analysts_{job_id}"]
                if hasattr(vector_data, "metadata") and "analysts" in vector_data.metadata:
                    # Deserialize JSON
                    analysts_json = vector_data.metadata["analysts"]
                    return json.loads(analysts_json)
            
            return None
        except Exception as e:
            print(f"Error retrieving analysts: {e}")
            return None

# Initialize API with required components
def create_api():
    # Import your research components
    from set_env import setup_environment
    from analysts import create_analysts
    from build_graph import build_graph
    import converse
    import finalize
    from langchain_community.tools.tavily_search import TavilySearchResults
    
    # Initialize API
    fastapi_app = FastAPI(title="Research Assistant API")
    
    # Initialize Pinecone job tracker
    job_tracker = PineconeJobTracker(
        api_key=os.environ.get("PINECONE_API_KEY"),
        index_name="langgraph-jobs"
    )
    
    # Set up LLM and tools
    llm = setup_environment()
    converse.llm = llm
    finalize.llm = llm
    converse.tavily_search = TavilySearchResults(max_results=3)
    
    # Create the graph
    create_analysts_with_llm = lambda state: create_analysts(state, llm)
    graph = build_graph(create_analysts_with_llm)
    
    # API endpoints
    @fastapi_app.post("/api/research", response_model=ResearchResponse)
    async def create_research(request: ResearchRequest, background_tasks: BackgroundTasks):
        # Generate job ID
        job_id = f"job_{uuid.uuid4().hex[:8]}"
        
        # Create job in Pinecone
        job_tracker.create_job(job_id, request.topic, request.num_analysts)
        
        # Start research in background
        background_tasks.add_task(
            run_research_job, 
            job_id, 
            request.topic, 
            request.num_analysts, 
            graph,
            job_tracker
        )
        
        return {"job_id": job_id, "status": "starting"}
    
    @fastapi_app.get("/api/research/test")
    async def test_api():
        """Simple endpoint to test if the API is running"""
        return {"status": "ok", "message": "API is working"}

    @fastapi_app.get("/api/research/{job_id}", response_model=ResearchStatus)
    async def get_research_status(job_id: str):
        # Get job from Pinecone
        job_data = job_tracker.get_job(job_id)
        
        if not job_data:
            raise HTTPException(status_code=404, detail="Job not found")
        
        # Get analysts (either from job data or dedicated vector)
        analysts = job_data.get("analysts")
        if not analysts:
            analysts = job_tracker.get_analysts(job_id)
            
        # Set up response
        response = {
            "job_id": job_id,
            "status": job_data.get("status", "unknown"),
            "progress": float(job_data.get("progress", 0.0)),
            "current_step": job_data.get("current_step", ""),
            "analysts": analysts
        }
        
        # Get final report if job is complete
        if job_data.get("status") == "completed":
            report = job_tracker.get_report(job_id)
            if report:
                response["final_report"] = report
            elif "final_report" in job_data:
                response["final_report"] = job_data["final_report"]
            elif "report" in job_data:
                response["final_report"] = job_data["report"]
            
            # Include error message if any
            if job_data.get("status") == "error" and "error" in job_data:
                response["error"] = job_data["error"]
        
        return response
    
    # Function to run research
    def run_research_job(job_id, topic, num_analysts, graph, job_tracker):
        try:
            # Update status
            job_tracker.update_job(job_id, {"status": "running"})
            
            # Initial state
            initial_state = {
                'topic': topic, 
                'number_analysts': num_analysts,
                'sections': []
            }
            
            # Thread ID for graph
            thread_id = f"research_{topic.replace(' ', '_')}_{uuid.uuid4().hex[:8]}"
            thread = {'configurable': {'thread_id': thread_id}}
            
            # Track progress through graph execution
            for event in graph.stream(initial_state, thread, stream_mode="updates"):
                node_name = next(iter(event.keys()))
                
                # Update progress based on node
                if node_name == "create_analysts":
                    if 'analysts' in event[node_name]:
                        # Convert analysts to a dictionary format that can be serialized
                        analysts_data = [
                            {k: v for k, v in (analyst.model_dump() if hasattr(analyst, 'model_dump') else analyst.dict()).items()} 
                            for analyst in event[node_name]['analysts']
                        ]
                        
                        # Print for debugging
                        print(f"Saving {len(analysts_data)} analysts to job {job_id}")
                        
                        # First save analysts to job state
                        job_tracker.update_job(job_id, {
                            "analysts": analysts_data,
                            "progress": 0.2,
                            "current_step": "Conducting interviews",
                            "analyst_count": len(analysts_data)
                        })
                        
                        # For extra reliability, save analysts in their own vector
                        try:
                            job_tracker.index.upsert(
                                vectors=[{
                                    "id": f"analysts_{job_id}",
                                    "values": [0.0] * 1535 + [1.0],  # Non-zero vector
                                    "metadata": {"analysts": json.dumps(analysts_data)}
                                }],
                                namespace="job_analysts"
                            )
                            print(f"Saved analysts in separate vector")
                        except Exception as e:
                            print(f"Error saving analysts in separate vector: {e}")
                
                elif node_name == "write_report":
                    job_tracker.update_job(job_id, {
                        "progress": 0.6,
                        "current_step": "Writing report"
                    })
                
                elif node_name == "write_introduction":
                    job_tracker.update_job(job_id, {
                        "progress": 0.7,
                        "current_step": "Writing introduction"
                    })
                
                elif node_name == "write_conclusion":
                    job_tracker.update_job(job_id, {
                        "progress": 0.8,
                        "current_step": "Writing conclusion"
                    })
                
                elif node_name == "finalize_report":
                    if 'final_report' in event[node_name]:
                        # Save the report to Pinecone
                        job_tracker.save_report(job_id, event[node_name]['final_report'])
                        
                        # Update job status
                        job_tracker.update_job(job_id, {
                            "progress": 1.0,
                            "current_step": "Complete",
                            "status": "completed",
                            "completed_at": str(datetime.datetime.now())
                        })
            
            # Ensure completion
            job_status = job_tracker.get_job(job_id)
            if job_status and float(job_status.get("progress", 0)) < 1.0:
                job_tracker.update_job(job_id, {
                    "progress": 1.0,
                    "current_step": "Complete",
                    "status": "completed",
                    "completed_at": str(datetime.datetime.now())
                })
                
        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            print(f"Error in research job: {error_details}")
            # Update job status on error
            job_tracker.update_job(job_id, {
                "status": "error",
                "error": str(e)
            })
    
    # Return the FastAPI app
    return fastapi_app

# Create app instance - this is critical for uvicorn to find it
app = create_api()

if __name__ == "__main__":
    import uvicorn
    
    # Debug environment variables
    print(f"PINECONE_API_KEY exists: {bool(os.environ.get('PINECONE_API_KEY'))}")
    print(f"OPENAI_API_KEY exists: {bool(os.environ.get('OPENAI_API_KEY'))}")
    
    # Set up environment variables if not already set
    os.environ.setdefault("STORAGE_TYPE", "pinecone")
    os.environ.setdefault("PINECONE_INDEX", "langgraph_checkpoints")
    
    # Run the app
    uvicorn.run(app, host="0.0.0.0", port=8000)