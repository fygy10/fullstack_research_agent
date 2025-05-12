# import os
# import json
# import pickle
# import base64
# from pinecone import Pinecone, ServerlessSpec
# from langgraph.checkpoint.base import BaseCheckpointSaver



# class PineconeSaver(BaseCheckpointSaver):

#     """
#     Pinecone implementation of checkpoint saver for LangGraph.
#     """
    
#     #pinecone params
#     def __init__(
#         self,
#         api_key: str = None,
#         index_name: str = "langgraph_checkpoints",
#         namespace: str = "checkpoints",
#         dimension: int = 1536,
#         serializer="pickle"
#     ):

#         #API key from .env 
#         self.api_key = api_key or os.environ.get("PINECONE_API_KEY")
#         if not self.api_key:
#             raise ValueError("Pinecone API key not provided and PINECONE_API_KEY env var not set")
        
#         #sanitize for pinecone requirements
#         self.index_name = index_name.lower().replace('_', '-')
#         self.namespace = namespace
#         self.dimension = dimension
#         self.serializer = serializer
        

#         #initialize Pinecone client
#         self.pc = Pinecone(api_key=self.api_key)
        

#         #check if index (vector database) exists, create if not
#         try:
#             indexes = self.pc.list_indexes()
#             index_exists = any(idx.name == self.index_name for idx in indexes)
            
#             #create index and its environment in cloud 
#             if not index_exists:
#                 print(f"Creating Pinecone index: {self.index_name}")
#                 self.pc.create_index(
#                     name=self.index_name,
#                     dimension=self.dimension,
#                     metric="cosine",
#                     spec=ServerlessSpec(cloud="aws", region="us-east-1")
#                 )
                
#             #connect to the index
#             self.index = self.pc.Index(self.index_name)
            
#             #test connection
#             stats = self.index.describe_index_stats()
#             print(f"Connected to Pinecone index: {self.index_name}")
#             print(f"Index stats: {stats}")
#         except Exception as e:
#             raise ConnectionError(f"Failed to connect to Pinecone index: {e}")
    


#     #convert data to a string for storage
#     def _serialize(self, data):

#         """Serialize data to string."""

#         #default json, back-up pickle
#         try:
#             if self.serializer == "json":
#                 return json.dumps(data)
#             else:
#                 serialized = pickle.dumps(data)
#                 return base64.b64encode(serialized).decode('utf-8')
#         except Exception as e:
#             print(f"Serialization error: {e}")
#             serialized = pickle.dumps(str(data))
#             return base64.b64encode(serialized).decode('utf-8')
    


#     #converrt data back to an object
#     def _deserialize(self, data_str):

#         """Deserialize data from string."""

#         if not data_str:
#             return None

#         #default json, back-up pickle
#         try:
#             if self.serializer == "json":
#                 return json.loads(data_str)
#             else:
#                 binary_data = base64.b64decode(data_str)
#                 return pickle.loads(binary_data)
#         except Exception as e:
#             print(f"Deserialization error: {e}")
#             return None
    

#     #package dummy vector with serialized strings
#     def _create_vector(self, thread_id, data, metadata=None, versions=None):

#         """Create a vector representation of the checkpoint data."""

#         #create a dummy vector (zeros)
#         dummy_vector = [0.0] * self.dimension
        
#         #serialize the data
#         serialized_data = self._serialize(data)
            
#         #create metadata object
#         meta = {
#             "data": serialized_data,
#             "serializer": self.serializer
#         }
        
#         #add versions if provided
#         if versions:
#             meta["versions"] = json.dumps(versions)
            
#         #add custom metadata if provided
#         if metadata:
#             meta["custom_metadata"] = json.dumps(metadata)
            
#         return {
#             "id": f"thread_{thread_id}",
#             "values": dummy_vector,
#             "metadata": meta
#         }

        
#     #retrieve checkpoint data from pinecone
#     def get(self, thread_id):

#         """Get checkpoint for a thread."""

#         #extract thread_id from config if it is a dict
#         if isinstance(thread_id, dict):
#             config = thread_id
#             thread_id = config.get("thread_id")
#             if not thread_id and "configurable" in config:
#                 thread_id = config.get("configurable", {}).get("thread_id")
            
#             if not thread_id:
#                 return None
        

#         try:
#             #query the vector by ID
#             response = self.index.fetch(
#                 ids=[f"thread_{thread_id}"],
#                 namespace=self.namespace
#             )
            
#             #check if we got a result
#             vectors = response.get("vectors", {})
#             if not vectors or f"thread_{thread_id}" not in vectors:
#                 return None
                
#             #extract the metadata from vector data
#             vector_data = vectors[f"thread_{thread_id}"]
#             metadata = vector_data.get("metadata", {})
            
#             #get the serialized data from vector data
#             serialized_data_str = metadata.get("data")
#             if not serialized_data_str:
#                 return None
                
#             #deserialize back to object
#             return self._deserialize(serialized_data_str)


#         except Exception as e:
#             print(f"Error retrieving data from Pinecone: {e}")
#             return None
    

#     #write the checkpoint data
#     def put(self, config, values, metadata=None, versions=None):

#         """Save checkpoint for a thread."""

#         #extract thread_id from config
#         thread_id = config.get("thread_id")
#         if not thread_id and "configurable" in config:
#             thread_id = config.get("configurable", {}).get("thread_id")
        
#         if not thread_id:
#             print("WARNING: No thread_id found in config, using default")
#             thread_id = "default_thread"
            
#         #create vector data
#         vector_data = self._create_vector(thread_id, values, metadata, versions)
        
        
#         #push to Pinecone
#         try:
#             self.index.upsert(
#                 vectors=[vector_data],
#                 namespace=self.namespace
#             )
#             print(f"Saved checkpoint for thread {thread_id}")

#         except Exception as e:
#             print(f"Error saving data to Pinecone: {e}")
    


#     #show all checkpoints if called
#     def list(self):

#         """List all thread IDs with checkpoints."""

#         #top 10KK results
#         try:
#             response = self.index.query(
#                 vector=[0.0] * self.dimension,
#                 top_k=10000,
#                 include_metadata=False,
#                 namespace=self.namespace
#             )
            
#             #return all thread ids
#             thread_ids = []
#             for match in response.get("matches", []):
#                 thread_id = match.get("id", "").replace("thread_", "")
#                 if thread_id:
#                     thread_ids.append(thread_id)
                    
#             return thread_ids
            
#         except Exception as e:
#             print(f"Error listing threads from Pinecone: {e}")
#             return []
    

#     #remove checkpoint data
#     def delete(self, thread_id):

#         """Delete checkpoint for a thread."""

#         #delete logic
#         try:
#             self.index.delete(
#                 ids=[f"thread_{thread_id}"],
#                 namespace=self.namespace
#             )
#             print(f"Deleted checkpoint for thread {thread_id}")

#         except Exception as e:
#             print(f"Error deleting data from Pinecone: {e}")
    


#     #CHECKPOINTS + VERSION DATA

#     def get_tuple(self, config):

#         """Get checkpoint and version for a thread."""

#         # Extract thread_id from config
#         thread_id = config.get("thread_id")
#         if not thread_id and "configurable" in config:
#             thread_id = config.get("configurable", {}).get("thread_id")
        
#         if not thread_id:
#             return None
            
#         try:
#             # Query the vector by ID
#             response = self.index.fetch(
#                 ids=[f"thread_{thread_id}"],
#                 namespace=self.namespace
#             )
            
#             # Check if we got a result
#             vectors = response.get("vectors", {})
#             if not vectors or f"thread_{thread_id}" not in vectors:
#                 return None
                
#             # Extract the metadata
#             vector_data = vectors[f"thread_{thread_id}"]
#             metadata = vector_data.get("metadata", {})
            
#             # Get the serialized data
#             serialized_data_str = metadata.get("data")
#             if not serialized_data_str:
#                 return None
                
#             # Get versions if available
#             versions_str = metadata.get("versions")
#             versions = json.loads(versions_str) if versions_str else {}
                
#             # Deserialize the data
#             data = self._deserialize(serialized_data_str)
            
#             return data, versions
                
#         except Exception as e:
#             print(f"Error retrieving data from Pinecone: {e}")
#             return None
        
#     def put_tuple(self, config, values, versions):
#         """Save checkpoint and version for a thread."""
#         self.put(config, values, None, versions)
    
#     def put_writes(self, config, values, metadata=None, old_versions=None, new_versions=None):
#         """Save checkpoint with write version tracking."""
#         self.put(config, values, metadata, new_versions)
    
#     def get_with_meta(self, config):
#         """Get checkpoint, metadata, and versions for a thread."""
#         # Extract thread_id from config
#         thread_id = config.get("thread_id")
#         if not thread_id and "configurable" in config:
#             thread_id = config.get("configurable", {}).get("thread_id")
        
#         if not thread_id:
#             return None, None, None
            
#         try:
#             # Query the vector by ID
#             response = self.index.fetch(
#                 ids=[f"thread_{thread_id}"],
#                 namespace=self.namespace
#             )
            
#             # Check if we got a result
#             vectors = response.get("vectors", {})
#             if not vectors or f"thread_{thread_id}" not in vectors:
#                 return None, None, None
                
#             # Extract the metadata
#             vector_data = vectors[f"thread_{thread_id}"]
#             metadata = vector_data.get("metadata", {})
            
#             # Get the serialized data
#             serialized_data_str = metadata.get("data")
#             if not serialized_data_str:
#                 return None, None, None
                
#             # Get versions if available
#             versions_str = metadata.get("versions")
#             versions = json.loads(versions_str) if versions_str else {}
            
#             # Get custom metadata if available
#             custom_metadata_str = metadata.get("custom_metadata")
#             custom_metadata = json.loads(custom_metadata_str) if custom_metadata_str else None
                
#             # Deserialize the data
#             data = self._deserialize(serialized_data_str)
            
#             return data, custom_metadata, versions
                
#         except Exception as e:
#             print(f"Error retrieving data from Pinecone: {e}")
#             return None, None, None


from typing import Any, Dict, List, Optional, Tuple
import json
import pickle
import os
import base64
from langgraph.checkpoint.base import BaseCheckpointSaver

class PineconeSaver(BaseCheckpointSaver):
    """
    Pinecone implementation of checkpoint saver for LangGraph.
    Stores graph state checkpoints in Pinecone.
    """
    
    def __init__(
        self,
        api_key: str = None,
        index_name: str = "langgraph_checkpoints",
        namespace: str = "checkpoints",
        dimension: int = 1536,
        serializer="pickle"
    ):
        """
        Initialize the Pinecone checkpoint saver.
        
        Args:
            api_key: Pinecone API key (defaults to PINECONE_API_KEY env var)
            index_name: Name of the Pinecone index to use
            namespace: Namespace within the index for checkpoints
            dimension: Vector dimension for the index
            serializer: Serialization format - 'json' or 'pickle'
        """
        # Use API key from env var if not provided
        self.api_key = api_key or os.environ.get("PINECONE_API_KEY")
        if not self.api_key:
            raise ValueError("Pinecone API key not provided and PINECONE_API_KEY env var not set")
        
        self.index_name = index_name.lower().replace('_', '-')
        self.namespace = namespace
        self.dimension = dimension
        
        if serializer not in ["json", "pickle"]:
            raise ValueError("Serializer must be 'json' or 'pickle'")
        self.serializer = serializer
        
        # Initialize Pinecone client with new API
        from pinecone import Pinecone, ServerlessSpec
        self.pc = Pinecone(api_key=self.api_key)
        
        # Check if index exists, create if it doesn't
        # In pinecone_saver.py, modify the index creation part
# Find the try/except block for index creation and update it:

        try:
            indexes = self.pc.list_indexes()
            index_names = [idx.name for idx in indexes]
            index_exists = self.index_name in index_names
            
            if not index_exists:
                print(f"Creating Pinecone index: {self.index_name}")
                try:
                    self.pc.create_index(
                        name=self.index_name,
                        dimension=self.dimension,
                        metric="cosine",
                        spec=ServerlessSpec(
                            cloud="aws",
                            region="us-west-2"
                        )
                    )
                except Exception as e:
                    # Check if it's just a "already exists" error (409)
                    if hasattr(e, 'status') and e.status == 409:
                        print(f"Index {self.index_name} already exists, continuing...")
                    else:
                        raise  # Re-raise if it's a different error
            
            # Connect to the index (whether new or existing)
            self.index = self.pc.Index(self.index_name)
            
            # Test connection
            stats = self.index.describe_index_stats()
            print(f"Connected to Pinecone index: {self.index_name}")
            print(f"Index stats: {stats}")
        except Exception as e:
            raise ConnectionError(f"Failed to connect to Pinecone index: {e}")
        
    def _serialize(self, data):
        """Serialize data to string."""
        try:
            if self.serializer == "json":
                return json.dumps(data)
            else:
                serialized = pickle.dumps(data)
                return base64.b64encode(serialized).decode('utf-8')
        except Exception as e:
            print(f"Serialization error: {e}")
            # Fallback to simpler serialization
            serialized = pickle.dumps(str(data))
            return base64.b64encode(serialized).decode('utf-8')
    
    def _deserialize(self, data_str: str) -> Any:
        """Deserialize data from string."""
        if not data_str:
            return None
        try:
            if self.serializer == "json":
                return json.loads(data_str)
            else:
                binary_data = base64.b64decode(data_str)
                return pickle.loads(binary_data)
        except Exception as e:
            print(f"Deserialization error: {e}")
            return None
    
# In pinecone_saver.py, modify the _create_vector_from_data method to compress large data

    def _create_vector_from_data(self, thread_id, data, metadata=None, versions=None):
        """Create a vector representation of the checkpoint data."""
        # Create a dummy vector with at least one non-zero value
        dummy_vector = [0.0] * (self.dimension - 1) + [1.0]
        
        # Serialize the data
        serialized_data = self._serialize(data)
        
        # Split data if it's too large for Pinecone's 40KB metadata limit
        # Each vector can only hold ~40KB of metadata
        try:
            # Check if serialized data is too large
            serialized_data_size = len(serialized_data.encode('utf-8'))
            max_size = 35000  # Leave some room for other metadata fields
            
            if serialized_data_size > max_size:
                # Store main chunk
                primary_meta = {
                    "data_part": "primary",
                    "serializer": self.serializer,
                    "total_parts": 1,  # Will be updated if we split
                    "is_split": True
                }
                
                # Store versions if provided
                if versions:
                    try:
                        versions_json = json.dumps(versions)
                        if len(versions_json) < 1000:  # Small enough to include
                            primary_meta["versions"] = versions_json
                    except:
                        pass
                
                # Split data into chunks of max_size bytes
                chunks = []
                current_pos = 0
                while current_pos < len(serialized_data):
                    # Calculate end position based on byte size, not characters
                    end_pos = current_pos
                    while end_pos < len(serialized_data):
                        chunk = serialized_data[current_pos:end_pos+1]
                        if len(chunk.encode('utf-8')) > max_size:
                            break
                        end_pos += 1
                    
                    # If we couldn't fit even a single character, take at least something
                    if end_pos == current_pos:
                        end_pos = current_pos + 100  # Take at least 100 chars
                    
                    chunks.append(serialized_data[current_pos:end_pos])
                    current_pos = end_pos
                
                # Update total parts
                primary_meta["total_parts"] = len(chunks)
                primary_meta["data"] = chunks[0]  # First chunk goes in primary
                
                # Create and store additional chunks if needed
                if len(chunks) > 1:
                    for i, chunk in enumerate(chunks[1:], 1):
                        # Create auxiliary vector for this chunk
                        aux_id = f"thread_{thread_id}_part_{i}"
                        aux_meta = {
                            "data_part": f"part_{i}",
                            "data": chunk,
                            "parent_id": f"thread_{thread_id}"
                        }
                        
                        # Store auxiliary vector
                        aux_vector = {
                            "id": aux_id,
                            "values": dummy_vector,
                            "metadata": aux_meta
                        }
                        
                        try:
                            self.index.upsert(
                                vectors=[aux_vector],
                                namespace=f"{self.namespace}_parts"
                            )
                        except Exception as e:
                            print(f"Error storing part {i} data: {e}")
                
                # Return the primary vector
                return {
                    "id": f"thread_{thread_id}",
                    "values": dummy_vector,
                    "metadata": primary_meta
                }
            
            # If data fits in a single vector, use normal approach
            meta = {
                "data": serialized_data,
                "serializer": self.serializer,
                "is_split": False
            }
            
            # Add versions if provided
            if versions:
                try:
                    versions_json = json.dumps(versions)
                    if len(versions_json) < 1000:  # Small enough to include
                        meta["versions"] = versions_json
                except:
                    pass
                    
            # Add custom metadata if provided
            if metadata:
                try:
                    # Try converting metadata to JSON string and check size
                    meta_json = json.dumps(metadata)
                    if len(meta_json) < 4000:  # Only include if small enough
                        meta["custom_metadata"] = meta_json
                    else:
                        print(f"Warning: Metadata for thread {thread_id} too large ({len(meta_json)} bytes), skipping")
                except TypeError:
                    # If serialization fails, use a small placeholder
                    meta["custom_metadata"] = f"Metadata size: {len(str(metadata))} bytes (too large to store)"
            
            return {
                "id": f"thread_{thread_id}",
                "values": dummy_vector,
                "metadata": meta
            }
        except Exception as e:
            # Fallback to minimum metadata if everything else fails
            print(f"Error creating vector metadata: {e}, using minimal metadata")
            return {
                "id": f"thread_{thread_id}",
                "values": dummy_vector,
                "metadata": {
                    "data": "ERROR: Data too large to store",
                    "serializer": self.serializer,
                    "error": str(e)
                }
            }
        
# In pinecone_saver.py, update the get method to handle split data

    def get(self, thread_id):
        """Get checkpoint for a thread."""
        # Extract thread_id from config if it is a dict
        if isinstance(thread_id, dict):
            config = thread_id
            thread_id = config.get("thread_id")
            if not thread_id and "configurable" in config:
                thread_id = config.get("configurable", {}).get("thread_id")
            
            if not thread_id:
                return None

        try:
            # Query the vector by ID
            response = self.index.fetch(
                ids=[f"thread_{thread_id}"],
                namespace=self.namespace
            )
            
            # Check if we got a result
            if not hasattr(response, 'vectors') or f"thread_{thread_id}" not in response.vectors:
                return None
                
            # Extract the metadata
            vector_data = response.vectors[f"thread_{thread_id}"]
            metadata = vector_data.metadata if hasattr(vector_data, 'metadata') else {}
            
            # Check if data is split across multiple vectors
            is_split = metadata.get("is_split", False)
            
            if is_split:
                # Get the number of parts
                total_parts = metadata.get("total_parts", 1)
                
                # Start with the data from the primary vector
                combined_data = metadata.get("data", "")
                
                # Fetch additional parts if needed
                if total_parts > 1:
                    # Prepare IDs for all parts
                    part_ids = [f"thread_{thread_id}_part_{i}" for i in range(1, total_parts)]
                    
                    try:
                        # Fetch all additional parts
                        parts_response = self.index.fetch(
                            ids=part_ids,
                            namespace=f"{self.namespace}_parts"
                        )
                        
                        # Process each part in order
                        for i in range(1, total_parts):
                            part_id = f"thread_{thread_id}_part_{i}"
                            if hasattr(parts_response, 'vectors') and part_id in parts_response.vectors:
                                part_vector = parts_response.vectors[part_id]
                                part_data = part_vector.metadata.get("data", "")
                                combined_data += part_data
                            else:
                                print(f"Warning: Missing part {i} for thread {thread_id}")
                    except Exception as e:
                        print(f"Error fetching additional parts: {e}")
                
                # Deserialize the combined data
                return self._deserialize(combined_data)
            else:
                # Get the serialized data from single vector
                serialized_data_str = metadata.get("data")
                if not serialized_data_str:
                    return None
                
                # Deserialize
                return self._deserialize(serialized_data_str)

        except Exception as e:
            print(f"Error retrieving data from Pinecone: {e}")
            return None
    
    def put(self, config, values, metadata=None, versions=None):
        """
        Save checkpoint for a thread.
        
        Args:
            config: Checkpoint configuration containing thread_id
            values: The checkpoint data to save
            metadata: Optional metadata to save
            versions: Optional version information
        """
        # Extract thread_id from config
        thread_id = config.get("thread_id")
        if not thread_id and "configurable" in config:
            thread_id = config.get("configurable", {}).get("thread_id")
        
        if not thread_id:
            print("WARNING: No thread_id found in config, using default")
            thread_id = "default_thread"
            
        # Create vector data
        vector_data = self._create_vector_from_data(thread_id, values, metadata, versions)
        
        # Upsert to Pinecone
        try:
            self.index.upsert(
                vectors=[vector_data],
                namespace=self.namespace
            )
            print(f"Saved checkpoint for thread {thread_id}")
        except Exception as e:
            print(f"Error saving data to Pinecone: {e}")
    
    def list(self) -> List[str]:
        """
        List all thread IDs with checkpoints.
        
        Returns:
            List of thread IDs
        """
        try:
            # Query all vectors in the namespace
            # Since Pinecone doesn't have a "list all" function, we use a query with limit
            response = self.index.query(
                vector=[0.0] * self.dimension,  # Dummy vector
                top_k=10000,  # Adjust this based on expected number of checkpoints
                include_metadata=False,
                namespace=self.namespace
            )
            
            # Extract thread IDs from the matches
            thread_ids = []
            for match in response.get("matches", []):
                # Remove the "thread_" prefix
                thread_id = match.get("id", "").replace("thread_", "")
                if thread_id:
                    thread_ids.append(thread_id)
                    
            return thread_ids
            
        except Exception as e:
            print(f"Error listing threads from Pinecone: {e}")
            return []
    
    def delete(self, thread_id: str) -> None:
        """
        Delete checkpoint for a thread.
        
        Args:
            thread_id: The thread ID to delete checkpoint for
        """
        try:
            self.index.delete(
                ids=[f"thread_{thread_id}"],
                namespace=self.namespace
            )
            print(f"Deleted checkpoint for thread {thread_id}")
        except Exception as e:
            print(f"Error deleting data from Pinecone: {e}")
    
    def get_tuple(self, config: Dict[str, Any]) -> Optional[Tuple[Dict[str, Any], Dict[str, int]]]:
        """
        Get checkpoint and version for a thread.
        
        Args:
            config: Checkpoint configuration containing thread_id
            
        Returns:
            Tuple of (checkpoint data, versions) or None if not found
        """
        # Extract thread_id from config
        thread_id = config.get("thread_id")
        if not thread_id and "configurable" in config:
            thread_id = config.get("configurable", {}).get("thread_id")
        
        if not thread_id:
            return None
            
        try:
            # Query the vector by ID
            response = self.index.fetch(
                ids=[f"thread_{thread_id}"],
                namespace=self.namespace
            )
            
            # Check if we got a result
            if not response.get("vectors") or f"thread_{thread_id}" not in response.get("vectors", {}):
                return None
                
            # Extract the metadata
            vector_data = response["vectors"][f"thread_{thread_id}"]
            metadata = vector_data.get("metadata", {})
            
            # Get the serialized data
            serialized_data_str = metadata.get("data")
            if not serialized_data_str:
                return None
                
            # Get versions if available
            versions_str = metadata.get("versions")
            versions = json.loads(versions_str) if versions_str else {}
                
            # Deserialize the data
            data = self._deserialize(serialized_data_str)
            
            return data, versions
                
        except Exception as e:
            print(f"Error retrieving data from Pinecone: {e}")
            return None
        
    def put_tuple(self, config: Dict[str, Any], values: Dict[str, Any], versions: Dict[str, int]) -> None:
        """
        Save checkpoint and version for a thread.
        
        Args:
            config: Checkpoint configuration containing thread_id
            values: The checkpoint data to save
            versions: Version information
        """
        # Just call put with versions
        self.put(config, values, None, versions)
    
    # Implement the remaining methods required by BaseCheckpointSaver
    def put_writes(self, config: Dict[str, Any], values: Dict[str, Any], 
                  metadata: Optional[Dict[str, Any]] = None,
                  old_versions: Optional[Dict[str, int]] = None,
                  new_versions: Optional[Dict[str, int]] = None) -> None:
        """
        Save checkpoint with write version tracking.
        
        Args:
            config: Checkpoint configuration containing thread_id
            values: The checkpoint data to save
            metadata: Optional metadata to save
            old_versions: Previous version information
            new_versions: New version information
        """
        # Just use the new versions
        self.put(config, values, metadata, new_versions)
    
    def get_with_meta(self, config: Dict[str, Any]) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]], Optional[Dict[str, int]]]:
        """
        Get checkpoint, metadata, and versions for a thread.
        
        Args:
            config: Checkpoint configuration containing thread_id
            
        Returns:
            Tuple of (checkpoint data, metadata, versions) or (None, None, None) if not found
        """
        # Extract thread_id from config
        thread_id = config.get("thread_id")
        if not thread_id and "configurable" in config:
            thread_id = config.get("configurable", {}).get("thread_id")
        
        if not thread_id:
            return None, None, None
            
        try:
            # Query the vector by ID
            response = self.index.fetch(
                ids=[f"thread_{thread_id}"],
                namespace=self.namespace
            )
            
            # Check if we got a result
            if not response.get("vectors") or f"thread_{thread_id}" not in response.get("vectors", {}):
                return None, None, None
                
            # Extract the metadata
            vector_data = response["vectors"][f"thread_{thread_id}"]
            metadata = vector_data.get("metadata", {})
            
            # Get the serialized data
            serialized_data_str = metadata.get("data")
            if not serialized_data_str:
                return None, None, None
                
            # Get versions if available
            versions_str = metadata.get("versions")
            versions = json.loads(versions_str) if versions_str else {}
            
            # Get custom metadata if available
            custom_metadata_str = metadata.get("custom_metadata")
            custom_metadata = json.loads(custom_metadata_str) if custom_metadata_str else None
                
            # Deserialize the data
            data = self._deserialize(serialized_data_str)
            
            return data, custom_metadata, versions
                
        except Exception as e:
            print(f"Error retrieving data from Pinecone: {e}")
            return None, None, None