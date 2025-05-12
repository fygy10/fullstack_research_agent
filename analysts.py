from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from models import Analyst, Perspectives, GeneratedAnalystState

#template for analyst creation instructions
analyst_instructions = """
You are tasked with creating a set of AI analyst personas. Follow these instructions carefully:

1. First, review the research topic:
{topic} 

2. Create exactly {number_analysts} total analysts.

3. Each analyst should have a unique perspective and expertise that contributes valuable insights to the topic.

4. Design analysts with diverse backgrounds, areas of expertise, and viewpoints to provide comprehensive coverage.

5. Each analyst should focus on different aspects or implications of the research topic.
"""

#create analyst personas + sav to analyst state
def create_analysts(state: GeneratedAnalystState, llm):

    #parameters
    topic = state['topic']
    number_analysts = state['number_analysts']
    
    print(f"Creating {number_analysts} analysts to research{topic}")
    
    #structured output models
    structured_llm = llm.with_structured_output(Perspectives)
    
    #system message based on defined criteria above
    system_message = SystemMessage(content=analyst_instructions.format(
        topic=topic,
        number_analysts=number_analysts
    ))

    #generate question
    result = structured_llm.invoke([system_message, HumanMessage(content='Generate the set of analysts.')])

    #write analysts to state (response is in the form of the analyst class defined)
    return {'analysts': result.analysts}