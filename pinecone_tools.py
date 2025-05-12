# import os
# import pinecone
# from langchain_openai import OpenAIEmbeddings

# #initialize global variables - prevent immediate startup until needed
# pinecone_initialized = False
# embeddings = None


# #trigger initialization on first call
# def initialize_pinecone():

#     """Initialize Pinecone client and embeddings model."""

#     global pinecone_initialized, embeddings
    
#     #only initialize once
#     if pinecone_initialized:
#         return
    
#     #API key from environment
#     pinecone_api_key = os.environ.get('PINECONE_API_KEY')
#     if not pinecone_api_key:
#         raise ValueError("Pinecone API key not found. Set PINECONE_API_KEY environment variable.")
    
#     #initialize Pinecone client 
#     pinecone.init(api_key=pinecone_api_key)
    
#     #initialize OpenAI for embedding
#     openai_api_key = os.environ.get('OPENAI_API_KEY')

#     if not openai_api_key:
#         raise ValueError("OpenAI API key not found. Set OPENAI_API_KEY environment variable.")
    
#     embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    
#     pinecone_initialized = True
#     print("Pinecone and embeddings initialized successfully.")



# #vector similiart search in pinecone
# def search_pinecone(state, index_name, namespace, query, top_k=3):

#     """
#     Search Pinecone for similar vectors to the query.
    
#     Args:
#         state: The state object from LangGraph
#         index_name: The name of the Pinecone index to search
#         namespace: The namespace within the index
#         query: The search query string
#         top_k: Number of results to return
        
#     Returns:
#         Updated state with the search results added to context
#     """

#     #initialize if not already
#     initialize_pinecone()
    

#     try:
#         #get the index
#         index = pinecone.Index(index_name)
        
#         #create embeddings from the query
#         query_embedding = embeddings.embed_query(query)
        
#         #search Pinecone
#         results = index.query(
#             vector=query_embedding,
#             top_k=top_k,
#             include_metadata=True,
#             namespace=namespace
#         )
        
#         #format the results
#         formatted_results = []
#         for i, match in enumerate(results.get("matches", [])):
#             if match.get("metadata") and match["metadata"].get("text"):
#                 formatted_results.append(f"Result {i+1} (Score: {match['score']:.4f}):\n{match['metadata']['text']}\n")
        
#         #combine the results into a single string
#         combined_results = "\n".join(formatted_results)
        
#         #return updated state
#         return {"context": state.get("context", []) + [combined_results]}
    

#     except Exception as e:
#         print(f"Error searching Pinecone: {e}")
#         error_message = f"Error searching Pinecone: {str(e)}"
#         return {"context": state.get("context", []) + [error_message]}



# #pre-config search node
# def pinecone_tool_factory(index_name, namespace):

#     """
#     Create a configured Pinecone search tool function.
    
#     Args:
#         index_name: The name of the Pinecone index to search
#         namespace: The namespace within the index
        
#     Returns:
#         A function that takes state and query and returns updated state
#     """

#     def pinecone_search_tool(state):

#         """
#         Node function for searching Pinecone.
        
#         Args:
#             state: The state object from LangGraph
            
#         Returns:
#             Updated state with the search results
#         """

#         #the query from the most recent message
#         messages = state.get("messages", [])
#         if not messages:
#             return {"context": ["No messages found to extract query from."]}
        
#         #the last message (from the analyst)
#         last_message = messages[-1]
#         query = last_message.content
        
#         #search Pinecone
#         return search_pinecone(state, index_name, namespace, query)
    

#     return pinecone_search_tool



# #build knowledge base with docs passed in
# def populate_pinecone(file_path, index_name, namespace, chunk_size=1000, overlap=200):

#     """
#     Populate Pinecone with data from a file.
    
#     Args:
#         file_path: Path to the file to process
#         index_name: Name of the Pinecone index
#         namespace: Namespace within the index
#         chunk_size: Size of chunks to split text into
#         overlap: Overlap between chunks
#     """

#     #initialize Pinecone
#     initialize_pinecone()
    
#     #load file content
#     try:
#         with open(file_path, 'r', encoding='utf-8') as file:
#             content = file.read()

#     except Exception as e:
#         print(f"Error reading file {file_path}: {e}")
#         return
    
#     #split into chunks
#     chunks = chunk_text(content, chunk_size, overlap)
#     print(f"Split {file_path} into {len(chunks)} chunks")
    
#     #get or create index
#     indexes = pinecone.list_indexes()
#     if index_name not in indexes:
#         print(f"Creating index {index_name}")
#         pinecone.create_index(
#             name=index_name,
#             dimension=1536,  # Dimension for text-embedding-3-small
#             metric="cosine"
#         )
    
#     index = pinecone.Index(index_name)
    
#     #create and upsert embeddings
#     total_vectors = 0
#     for i, chunk in enumerate(chunks):

#         try:
#             #create embedding
#             embedding = embeddings.embed_query(chunk)
            
#             #upsert to Pinecone
#             index.upsert(
#                 vectors=[{
#                     "id": f"{os.path.basename(file_path)}_chunk_{i}",
#                     "values": embedding,
#                     "metadata": {
#                         "text": chunk,
#                         "source": file_path,
#                         "chunk_index": i
#                     }
#                 }],
#                 namespace=namespace
#             )
            
#             total_vectors += 1
#             if total_vectors % 10 == 0:
#                 print(f"Inserted {total_vectors} vectors...")
                

#         except Exception as e:
#             print(f"Error processing chunk {i} from {file_path}: {e}")
    
#     print(f"Completed! Inserted {total_vectors} vectors into Pinecone index {index_name}")
    
#     #get index stats
#     stats = index.describe_index_stats()
#     print(f"Index stats: {stats}")



# #break up large doc 
# def chunk_text(text, chunk_size=1000, overlap=200):

#     """Split text into chunks with overlap."""

#     if not text:
#         return []
    
#     chunks = []
#     start = 0
#     while start < len(text):
#         end = min(start + chunk_size, len(text))
#         # If not at the end, try to find a good break point
#         if end < len(text):
#             for break_char in ['. ', '\n', ' ']:
#                 last_break = text.rfind(break_char, start, end)
#                 if last_break != -1:
#                     end = last_break + 1
#                     break
        
#         chunks.append(text[start:end])
#         start = end - overlap
    
#     return chunks


import os
from langchain_openai import OpenAIEmbeddings
from pinecone import Pinecone, ServerlessSpec

# Initialize global variables - prevent immediate startup until needed
pinecone_initialized = False
embeddings = None
pc = None  # Pinecone client instance

# Trigger initialization on first call
def initialize_pinecone():
    """Initialize Pinecone client and embeddings model."""
    global pinecone_initialized, embeddings, pc
    
    # Only initialize once
    if pinecone_initialized:
        return
    
    # API key from environment
    pinecone_api_key = os.environ.get('PINECONE_API_KEY')
    if not pinecone_api_key:
        raise ValueError("Pinecone API key not found. Set PINECONE_API_KEY environment variable.")
    
    # Initialize Pinecone client with new API
    pc = Pinecone(api_key=pinecone_api_key)
    
    # Initialize OpenAI for embedding
    openai_api_key = os.environ.get('OPENAI_API_KEY')
    if not openai_api_key:
        raise ValueError("OpenAI API key not found. Set OPENAI_API_KEY environment variable.")
    
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    
    pinecone_initialized = True
    print("Pinecone and embeddings initialized successfully.")

# Vector similar search in pinecone
def search_pinecone(state, index_name, namespace, query, top_k=3):
    """
    Search Pinecone for similar vectors to the query.
    
    Args:
        state: The state object from LangGraph
        index_name: The name of the Pinecone index to search
        namespace: The namespace within the index
        query: The search query string
        top_k: Number of results to return
        
    Returns:
        Updated state with the search results added to context
    """
    # Initialize if not already
    initialize_pinecone()
    
    try:
        # Get the index with new API
        index = pc.Index(index_name)
        
        # Create embeddings from the query
        query_embedding = embeddings.embed_query(query)
        
        # Search Pinecone
        results = index.query(
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True,
            namespace=namespace
        )
        
        # Format the results
        formatted_results = []
        for i, match in enumerate(results.get("matches", [])):
            if match.get("metadata") and match["metadata"].get("text"):
                formatted_results.append(f"Result {i+1} (Score: {match['score']:.4f}):\n{match['metadata']['text']}\n")
        
        # Combine the results into a single string
        combined_results = "\n".join(formatted_results)
        
        # Return updated state
        return {"context": state.get("context", []) + [combined_results]}
    
    except Exception as e:
        print(f"Error searching Pinecone: {e}")
        error_message = f"Error searching Pinecone: {str(e)}"
        return {"context": state.get("context", []) + [error_message]}

# Pre-config search node
def pinecone_tool_factory(index_name, namespace):
    """
    Create a configured Pinecone search tool function.
    
    Args:
        index_name: The name of the Pinecone index to search
        namespace: The namespace within the index
        
    Returns:
        A function that takes state and query and returns updated state
    """
    def pinecone_search_tool(state):
        """
        Node function for searching Pinecone.
        
        Args:
            state: The state object from LangGraph
            
        Returns:
            Updated state with the search results
        """
        # The query from the most recent message
        messages = state.get("messages", [])
        if not messages:
            return {"context": ["No messages found to extract query from."]}
        
        # The last message (from the analyst)
        last_message = messages[-1]
        query = last_message.content
        
        # Search Pinecone
        return search_pinecone(state, index_name, namespace, query)
    
    return pinecone_search_tool

# Build knowledge base with docs passed in
def populate_pinecone(file_path, index_name, namespace, chunk_size=1000, overlap=200):
    """
    Populate Pinecone with data from a file.
    
    Args:
        file_path: Path to the file to process
        index_name: Name of the Pinecone index
        namespace: Namespace within the index
        chunk_size: Size of chunks to split text into
        overlap: Overlap between chunks
    """
    # Initialize Pinecone
    initialize_pinecone()
    
    # Load file content
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        return
    
    # Split into chunks
    chunks = chunk_text(content, chunk_size, overlap)
    print(f"Split {file_path} into {len(chunks)} chunks")
    
    # Get or create index
    indexes = pc.list_indexes()
    index_names = [idx.name for idx in indexes]
    
    if index_name not in index_names:
        print(f"Creating index {index_name}")
        pc.create_index(
            name=index_name,
            dimension=1536,  # Dimension for text-embedding-3-small
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region="us-west-2"
            )
        )
    
    index = pc.Index(index_name)
    
    # Create and upsert embeddings
    total_vectors = 0
    for i, chunk in enumerate(chunks):
        try:
            # Create embedding
            embedding = embeddings.embed_query(chunk)
            
            # Upsert to Pinecone with new API format
            index.upsert(
                vectors=[{
                    "id": f"{os.path.basename(file_path)}_chunk_{i}",
                    "values": embedding,
                    "metadata": {
                        "text": chunk,
                        "source": file_path,
                        "chunk_index": i
                    }
                }],
                namespace=namespace
            )
            
            total_vectors += 1
            if total_vectors % 10 == 0:
                print(f"Inserted {total_vectors} vectors...")
                
        except Exception as e:
            print(f"Error processing chunk {i} from {file_path}: {e}")
    
    print(f"Completed! Inserted {total_vectors} vectors into Pinecone index {index_name}")
    
    # Get index stats
    stats = index.describe_index_stats()
    print(f"Index stats: {stats}")

# Break up large doc 
def chunk_text(text, chunk_size=1000, overlap=200):
    """Split text into chunks with overlap."""
    if not text:
        return []
    
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        # If not at the end, try to find a good break point
        if end < len(text):
            for break_char in ['. ', '\n', ' ']:
                last_break = text.rfind(break_char, start, end)
                if last_break != -1:
                    end = last_break + 1
                    break
        
        chunks.append(text[start:end])
        start = end - overlap
    
    return chunks