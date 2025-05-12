# import os
# from dotenv import load_dotenv
# from langchain_openai import ChatOpenAI
# import mysql.connector


# def setup_environment():

#     load_dotenv()

#     #storage type set in .env
#     print(f"After load_dotenv, STORAGE_TYPE = {os.environ.get('STORAGE_TYPE')}")

#     #pinecone configuration
#     pinecone_api_key = os.environ.get('PINECONE_API_KEY', '')
#     pinecone_index = os.environ.get('PINECONE_INDEX', 'langgraph_checkpoints')


#     #set to storage type else default to in-memory
#     storage_type = os.environ.get('STORAGE_TYPE', 'memory')


#     #check Pinecone configuration if selected
#     if storage_type.lower() == 'pinecone':
#         if not pinecone_api_key:
#             print("Pinecone API key not found. Set PINECONE_API_KEY environment variable.")
#             print("Falling back to in-memory storage.")
#             os.environ['STORAGE_TYPE'] = 'memory'
    
#     # # PostgreSQL configuration
#     # postgres_host = os.environ.get('POSTGRES_HOST', 'localhost')
#     # postgres_user = os.environ.get('POSTGRES_USER', 'postgres')
#     # postgres_password = os.environ.get('POSTGRES_PASSWORD', '')
#     # postgres_db = os.environ.get('POSTGRES_DB', 'postgres')
#     # postgres_port = os.environ.get('POSTGRES_PORT', '5432')
    
#     # # Build connection string
#     # os.environ['POSTGRES_CONNECTION_STRING'] = f"postgresql://{postgres_user}:{postgres_password}@{postgres_host}:{postgres_port}/{postgres_db}"
    
#     # # Redis configuration
#     # redis_host = os.environ.get('REDIS_HOST', 'localhost')
#     # redis_port = os.environ.get('REDIS_PORT', '6379')
#     # redis_password = os.environ.get('REDIS_PASSWORD', '')
#     # redis_db = os.environ.get('REDIS_DB', '0')
    

#     # # Try to import Redis to test if it's available
#     # try:
#     #     import redis
#     #     # Test Redis connection if Redis storage is selected
#     #     if storage_type.lower() == 'redis':
#     #         try:
#     #             redis_client = redis.Redis(
#     #                 host=redis_host,
#     #                 port=int(redis_port),
#     #                 password=redis_password if redis_password else None,
#     #                 db=int(redis_db),
#     #                 socket_timeout=5
#     #             )
#     #             if redis_client.ping():
#     #                 print(f"Redis connection successful - connected to {redis_host}:{redis_port}")
#     #         except redis.exceptions.ConnectionError as err:
#     #             print(f"Redis connection error: {err}")
#     #             print("Warning: Redis connection failed. Falling back to alternative storage.")
#     #             os.environ['STORAGE_TYPE'] = 'memory'
#     # except ImportError:
#     #     if storage_type.lower() == 'redis':
#     #         print("Redis package not installed. Install with: pip install redis")
#     #         print("Falling back to alternative storage.")
#     #         os.environ['STORAGE_TYPE'] = 'memory'
    
#     # # Check MySQL connection
#     # mysql_config = {
#     #     'host': os.environ.get('MYSQL_HOST', 'localhost'),
#     #     'user': os.environ.get('MYSQL_USER', 'silverside_user'),
#     #     'password': os.environ.get('MYSQL_PASSWORD', 'silverside_password'),
#     #     'database': os.environ.get('MYSQL_DATABASE', 'silverside_db'),
#     #     'port': int(os.environ.get('MYSQL_PORT', 3306))
#     # }
    
#     # try:
#     #     # Test MySQL connection
#     #     conn = mysql.connector.connect(**mysql_config)
#     #     if conn.is_connected():
#     #         print(f"MySQL connection successful - connected to {mysql_config['host']}:{mysql_config['port']}")
#     #         conn.close()
#     # except mysql.connector.Error as err:
#     #     print(f"MySQL connection error: {err}")
#     #     print("Warning: MySQL connection failed. Data persistence will not work.")
    
#     #openAI API key
#     if not os.environ.get('OPENAI_API_KEY'):
#         raise ValueError("OPENAI_API_KEY not found. Please add it to your .env file.")
    
#     #langChain & tavily keys
#     if os.environ.get('LANGCHAIN_API_KEY'):
#         os.environ["LANGCHAIN_TRACING_V2"] = "false"
#         os.environ["LANGCHAIN_PROJECT"] = "langchain-academy"
#         print("LangChain tracing enabled.")
#         os.environ.get('TAVILY_API_KEY')
#     else:
#         print("LangChain API key not found. Tracing will be disabled.")
    
#     #create and return LLM
#     return ChatOpenAI(model='gpt-3.5-turbo', temperature=0, max_tokens=1000)


import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

def setup_environment():
    # Check if we're in Modal or local environment
    if not os.environ.get("MODAL_ENVIRONMENT"):
        # Only load .env file in local development
        load_dotenv()
        print("Loaded environment from .env file")
    
    print(f"STORAGE_TYPE = {os.environ.get('STORAGE_TYPE', 'memory')}")
    
    # Get Pinecone configuration for logging
    pinecone_api_key = os.environ.get('PINECONE_API_KEY', '')
    pinecone_index = os.environ.get('PINECONE_INDEX', 'langgraph_checkpoints')
    
    # Set storage type - prioritizing Pinecone
    storage_type = os.environ.get('STORAGE_TYPE', 'pinecone')
    
    # Check Pinecone configuration
    if storage_type.lower() == 'pinecone':
        if not pinecone_api_key:
            print("Pinecone API key not found. Set PINECONE_API_KEY environment variable.")
            print("Falling back to in-memory storage.")
            os.environ['STORAGE_TYPE'] = 'memory'
        else:
            print(f"Using Pinecone storage with index: {pinecone_index}")
    
    # Check OpenAI API key
    if not os.environ.get('OPENAI_API_KEY'):
        print("WARNING: OPENAI_API_KEY not found. LLM operations will fail.")
    
    # LangChain settings if available
    if os.environ.get('LANGCHAIN_API_KEY'):
        os.environ["LANGCHAIN_TRACING_V2"] = "false"
        os.environ["LANGCHAIN_PROJECT"] = "research-assistant"
        print("LangChain tracing enabled.")
    
    # Create and return LLM
    model = os.environ.get('OPENAI_MODEL', 'gpt-3.5-turbo')
    temperature = float(os.environ.get('OPENAI_TEMPERATURE', 0))
    max_tokens = int(os.environ.get('OPENAI_MAX_TOKENS', 1000))
    
    print(f"Initializing OpenAI LLM with model={model}, temperature={temperature}")
    
    return ChatOpenAI(model=model, temperature=temperature, max_tokens=max_tokens)