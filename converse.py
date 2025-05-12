import operator 
from typing import Annotated, List
from langgraph.graph import MessagesState
from pydantic import BaseModel, Field
from models import Analyst
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.document_loaders import WikipediaLoader
from langchain_core.messages import get_buffer_string, AIMessage, HumanMessage, SystemMessage


#set in main.py
llm = None
tavily_search = None  


#format interview state
class InterviewState(MessagesState):    #appends instead of overwrite
    max_num_turns: int
    context: Annotated[list, operator.add] 
    analyst: Analyst
    interview: str
    sections: list


#structure llm query
class SearchQuery(BaseModel):
    search_query: str = Field(None, description="Search query for retrieval")



question_instructions = """
You are an analyst tasked with interviewing an expert to learn about a specific topic. 
Your goal is boil down to interesting and specific insights related to your topic.

1. Interesting: Insights that people will find surprising or non-obvious.
        
2. Specific: Insights that avoid generalities and include specific examples from the expert.

Here is your topic of focus and set of goals: {goals}
        
Begin by introducing yourself using a name that fits your persona, and then ask your question.

Continue to ask questions to drill down and refine your understanding of the topic.
        
When you are satisfied with your understanding, complete the conversation with: "Thank you so much for your help!"

Remember to stay in character throughout your response, reflecting the persona and goals provided to you."""



search_instructions = SystemMessage(content=f"""You will be given a conversation between an analyst and an expert. 

Your goal is to generate a well-structured query for use in retrieval and / or web-search related to the conversation.
        
First, analyze the full conversation.

Pay particular attention to the final question posed by the analyst.

Convert this final question into a well-structured web search query""")



answer_instructions = """You are an expert being interviewed by an analyst.

Here is analyst area of focus: {goals}. 
        
You goal is to answer a question posed by the interviewer.

To answer question, use this context:
        
{context}

When answering questions, follow these guidelines:
        
1. Use only the information provided in the context. 
        
2. Do not introduce external information or make assumptions beyond what is explicitly stated in the context.

3. The context contain sources at the topic of each individual document.

4. Include these sources your answer next to any relevant statements. For example, for source # 1 use [1]. 

5. List your sources in order at the bottom of your answer. [1] Source 1, [2] Source 2, etc
        
6. If the source is: <Document source="assistant/docs/llama3_1.pdf" page="7"/>' then just list: 
        
[1] assistant/docs/llama3_1.pdf, page 7 
        
And skip the addition of the brackets as well as the Document source preamble in your citation."""



#generate questions with analyst
def generate_questions(state: InterviewState):

    #analyst and historical messages (qustion & answer) from interview state
    analyst = state["analyst"]
    messages = state["messages"]


    #question based on system message, analyst, and historical messages
    system_message = question_instructions.format(goals=analyst.persona)
    question = llm.invoke([SystemMessage(content=system_message)] + messages)
        
    #messages question added to state
    return {"messages": [question]}


#run web search
def search_web(state: InterviewState):

    #search query format + instructions with messages
    structured_llm = llm.with_structured_output(SearchQuery)
    search_query = structured_llm.invoke([search_instructions] + state['messages'])
    
    #Tavily search
    search_docs = tavily_search.invoke(search_query.search_query)

    #format
    formatted_search_docs = "\n\n---\n\n".join(
        [
            f'<Document href="{doc["url"]}"/>\n{doc["content"]}\n</Document>'
            for doc in search_docs
        ]
    )

    #add to the state
    return {"context": [formatted_search_docs]}


#wikipedia format + instructions with messages
def search_wikipedia(state: InterviewState):

    #search query
    structured_llm = llm.with_structured_output(SearchQuery)
    search_query = structured_llm.invoke([search_instructions] + state['messages'])
    
    #search
    search_docs = WikipediaLoader(query=search_query.search_query, load_max_docs=2).load()

    #format
    formatted_search_docs = "\n\n---\n\n".join(
        [
            f'<Document source="{doc.metadata["source"]}" page="{doc.metadata.get("page", "")}"/>\n{doc.page_content}\n</Document>'
            for doc in search_docs
        ]
    )

    return {"context": [formatted_search_docs]}


#generate answer from question responses
def generate_answer(state: InterviewState):

    #get state keys
    analyst = state["analyst"]
    messages = state["messages"]
    context = state["context"]

    #answer question format with system message + messages
    system_message = answer_instructions.format(goals=analyst.persona, context=context)
    answer = llm.invoke([SystemMessage(content=system_message)] + messages)
            
    #id message as coming from the expert
    answer.name = "expert"
    
    #append it to state
    return {"messages": [answer]}


#save interview to states
def save_interview(state: InterviewState):
    

    #get messages
    messages = state["messages"]
    
    #convert interview to a string
    interview = get_buffer_string(messages)
    
    #save to interviews 
    return {"interview": interview}



#router is run after each question - answer pair 
def route_messages(state: InterviewState, name: str = "expert"):
    
    #get state keys
    messages = state["messages"]
    max_num_turns = state.get('max_num_turns', 3)    

    #check the number of expert answers 
    num_responses = len(
        [m for m in messages if isinstance(m, AIMessage) and m.name == name]
    )

    #end if expert has answered more than the max turns
    if num_responses >= max_num_turns:
        return 'save_interview'

    #get the last question asked to check if it signals the end of discussion
    last_question = messages[-2]
    
    #nice to the llm so it doesn't hate us when we take over
    if "Thank you so much for your help" in last_question.content:
        return 'save_interview'
    return "ask_question"



section_writer_instructions = """You are an expert technical writer. 
            
Your task is to create a short, easily digestible section of a report based on a set of source documents.

1. Analyze the content of the source documents: 
- The name of each source document is at the start of the document, with the <Document tag.
        
2. Create a report structure using markdown formatting:
- Use ## for the section title
- Use ### for sub-section headers
        
3. Write the report following this structure:
a. Title (## header)
b. Summary (### header)
c. Sources (### header)

4. Make your title engaging based upon the focus area of the analyst: 
{focus}

5. For the summary section:
- Set up summary with general background / context related to the focus area of the analyst
- Emphasize what is novel, interesting, or surprising about insights gathered from the interview
- Create a numbered list of source documents, as you use them
- Do not mention the names of interviewers or experts
- Aim for approximately 400 words maximum
- Use numbered sources in your report (e.g., [1], [2]) based on information from source documents
        
6. In the Sources section:
- Include all sources used in your report
- Provide full links to relevant websites or specific document paths
- Separate each source by a newline. Use two spaces at the end of each line to create a newline in Markdown.
- It will look like:

### Sources
[1] Link or Document name
[2] Link or Document name

7. Be sure to combine sources. For example this is not correct:

[3] https://ai.meta.com/blog/meta-llama-3-1/
[4] https://ai.meta.com/blog/meta-llama-3-1/

There should be no redundant sources. It should simply be:

[3] https://ai.meta.com/blog/meta-llama-3-1/
        
8. Final review:
- Ensure the report follows the required structure
- Include no preamble before the title of the report
- Check that all guidelines have been followed"""



#create section based on interview results
def write_section(state: InterviewState):

    #get state keys
    analyst = state["analyst"]
    context = state["context"]
   
    #write section using docs + interview
    system_message = section_writer_instructions.format(focus=analyst.description)
    section = llm.invoke([SystemMessage(content=system_message)]+[HumanMessage(content=f"Use this source to write your section: {context}")]) 
                
    #add section to state
    return {"sections": [section.content]}