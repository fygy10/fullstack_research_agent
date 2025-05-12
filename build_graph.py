import os
from langgraph.graph import START, END, StateGraph
from langgraph.checkpoint.memory import MemorySaver
from models import GeneratedAnalystState
from langgraph.constants import Send
from finalize import ResearchGraphState, write_report, write_introduction, write_conclusion, finalize_report
from converse import InterviewState, generate_questions, search_web, search_wikipedia, generate_answer, save_interview, route_messages, write_section as write_section_func
from langchain_core.messages import HumanMessage
from pinecone_tools import pinecone_tool_factory
from pinecone_saver import PineconeSaver



#nest interview graph to run in parallel across all analysts
def build_interview_graph(use_pinecone=False, pinecone_index=None, pinecone_namespace=None):
    """
    Build the interview graph with optional Pinecone integration
    
    Args:
        use_pinecone: Whether to use Pinecone for vector search
        pinecone_index: Name of the Pinecone index to use
        pinecone_namespace: Namespace within the Pinecone index
    """
    #own state for interviews
    builder = StateGraph(InterviewState)
    
    #nodes with interview functions
    builder.add_node("ask_question", generate_questions)
    
    #conditionally add Pinecone search if requested
    if use_pinecone and pinecone_index and pinecone_namespace:
        pinecone_search = pinecone_tool_factory(pinecone_index, pinecone_namespace)
        builder.add_node("search_pinecone", pinecone_search)
        builder.add_edge("ask_question", "search_pinecone")
        builder.add_edge("search_pinecone", "search_web")
    else:
        builder.add_edge("ask_question", "search_web")
    
    builder.add_node("search_web", search_web)
    builder.add_node("search_wikipedia", search_wikipedia)
    builder.add_node("answer_question", generate_answer)
    builder.add_node("save_interview", save_interview)
    builder.add_node("write_section", write_section_func)
    
    #edges through the interview graph
    builder.add_edge(START, "ask_question")
    builder.add_edge("search_web", "search_wikipedia")
    builder.add_edge("search_wikipedia", "answer_question")
    builder.add_conditional_edges("answer_question", route_messages, {
        "ask_question": "ask_question",
        "save_interview": "save_interview"
    })
    builder.add_edge("save_interview", "write_section")
    builder.add_edge("write_section", END)
    
    return builder


#begins the interviews with the analysts
def initiate_all_interviews(state: ResearchGraphState):
    topic = state['topic']

    #allows to be run in parallel + append to main graph state
    return [Send('conduct_interview', {
        'analyst': analyst, 
        'messages': [HumanMessage(content=f'So you said you were writing a report on {topic}?')],
        'max_num_turns': 3, 
        'context': []
    }) for analyst in state['analysts']]


#main graph flow and functionality 
def build_graph(create_analysts_fn, use_pinecone=False, pinecone_config=None):

    """
    Build the main graph with optional Pinecone integration
    
    Args:
        create_analysts_fn: Function to create analyst personas
        use_pinecone: Whether to use Pinecone for storage and/or search
        pinecone_config: Configuration for Pinecone (api_key, index_name, namespace)
    """

    #default pinecone configuration if not provided specific pinecone config 
    if pinecone_config is None:
        pinecone_config = {
            'api_key': os.environ.get('PINECONE_API_KEY'),
            'index_name': os.environ.get('PINECONE_INDEX', 'langgraph_checkpoints'),
            'namespace': os.environ.get('PINECONE_NAMESPACE', 'research_checkpoints'),
            'search_namespace': os.environ.get('PINECONE_SEARCH_NAMESPACE', 'research_data')
        }
    
    #interview sub-graph with optional Pinecone search
    interview_builder = build_interview_graph(
        use_pinecone=use_pinecone,
        pinecone_index=pinecone_config.get('index_name'),
        pinecone_namespace=pinecone_config.get('search_namespace')
    )
    
    #main research graph
    builder = StateGraph(ResearchGraphState)
    
    #nodes
    builder.add_node("create_analysts", create_analysts_fn)
    builder.add_node("conduct_interview", interview_builder.compile())  #allows subgraph to be called and returned
    builder.add_node("write_report", write_report)
    builder.add_node("write_introduction", write_introduction)
    builder.add_node("write_conclusion", write_conclusion)
    builder.add_node("finalize_report", finalize_report)

    #edge flow
    builder.add_edge(START, "create_analysts")
    builder.add_conditional_edges("create_analysts", initiate_all_interviews)   #trigger for interviews with analysts
    builder.add_edge("conduct_interview", "write_report")
    builder.add_edge("conduct_interview", "write_introduction")
    builder.add_edge("conduct_interview", "write_conclusion")
    builder.add_edge(["write_conclusion", "write_report", "write_introduction"], "finalize_report")
    builder.add_edge("finalize_report", END)

    #storage backend selection - Redis or Pinecone
    print(f"In build_graph, STORAGE_TYPE environment: {os.environ.get('STORAGE_TYPE')}")
    storage_type = os.environ.get('STORAGE_TYPE', 'memory').lower()
    print(f"Selected storage_type: {storage_type}")
    
    # Add this to the try/except block in build_graph.py
    # Replace the existing try/except block:
    if storage_type == 'pinecone':
        try:
            # Import required modules
            from pinecone_saver import PineconeSaver
            # Import Pinecone with the correct import
            from pinecone import Pinecone
            
            print("Setting up Pinecone storage...")
            # Pinecone connection parameters
            pinecone_api_key = os.environ.get('PINECONE_API_KEY')
            pinecone_index = os.environ.get('PINECONE_INDEX', 'langgraph_checkpoints')
            
            print(f"Pinecone API key available: {bool(pinecone_api_key)}")
            print(f"Using Pinecone index: {pinecone_index}")
            
            # Create Pinecone saver
            try:
                pinecone_saver = PineconeSaver(
                    api_key=pinecone_api_key,
                    index_name=pinecone_index,
                    namespace="research_checkpoints"
                )
                print(f"Pinecone connection successful - connected to index: {pinecone_index}")
                return builder.compile(checkpointer=pinecone_saver)
            except Exception as e:
                import traceback
                print(f"Pinecone saver creation error: {str(e)}")
                print("Detailed traceback:")
                print(traceback.format_exc())
                raise
                
        except ImportError as e:
            print(f"Pinecone import error: {str(e)}")
            print("Pinecone module not available. Falling back to in-memory storage.")
        except Exception as e:
            import traceback
            print(f"Pinecone error: {str(e)}")
            print("Detailed traceback:")
            print(traceback.format_exc())
            print("Pinecone setup failed. Falling back to in-memory storage.")
    
    #Redis if available and configured
    elif storage_type == 'redis':
        try:
            import redis
            from langgraph.checkpoint.redis import RedisSaver
            
            redis_host = os.environ.get('REDIS_HOST', 'localhost')
            redis_port = os.environ.get('REDIS_PORT', '6379')
            redis_password = os.environ.get('REDIS_PASSWORD', '')
            
            print(f"Setting up Redis storage at {redis_host}:{redis_port}")
            
            redis_saver = RedisSaver(
                url=f"redis://{redis_host}:{redis_port}",
                password=redis_password if redis_password else None,
                ttl=3600  # 1 hour time-to-live
            )
            
            return builder.compile(checkpointer=redis_saver)
        except ImportError:
            print("Redis package not installed. Falling back to in-memory storage.")
        except Exception as e:
            print(f"Redis error: {str(e)}")
            print("Redis setup failed. Falling back to in-memory storage.")
    

    #fallback to memory storage if no other storage method worked
    print("Using in-memory storage")
    memory_saver = MemorySaver()
    return builder.compile(checkpointer=memory_saver)