import operator
from typing import List, Annotated 
from typing_extensions import TypedDict
from models import Analyst
from langchain_core.messages import HumanMessage, SystemMessage



#set in main.py
llm = None



#state structure
class ResearchGraphState(TypedDict):
    topic: str
    number_analysts: int
    analysts: List[Analyst]
    sections: Annotated[list, operator.add] #append not overwrite
    introduction: str
    content: str
    conclusion: str
    final_report: str



report_writer_instructions = """You are a technical writer creating a report on this overall topic: 

{topic}
    
You have a team of analysts. Each analyst has done two things: 

1. They conducted an interview with an expert on a specific sub-topic.
2. They write up their finding into a memo.

Your task: 

1. You will be given a collection of memos from your analysts.
2. Think carefully about the insights from each memo.
3. Consolidate these into a crisp overall summary that ties together the central ideas from all of the memos. 
4. Summarize the central points in each memo into a cohesive single narrative.

To format your report:
 
1. Use markdown formatting. 
2. Include no pre-amble for the report.
3. Use no sub-heading. 
4. Start your report with a single title header: ## Insights
5. Do not mention any analyst names in your report.
6. Preserve any citations in the memos, which will be annotated in brackets, for example [1] or [2].
7. Create a final, consolidated list of sources and add to a Sources section with the `## Sources` header.
8. List your sources in order and do not repeat.

[1] Source 1
[2] Source 2

Here are the memos from your analysts to build your report from: 

{context}"""



intro_conclusion_instructions = """You are a technical writer finishing a report on {topic}

You will be given all of the sections of the report.

You job is to write a crisp and compelling introduction or conclusion section.

The user will instruct you whether to write the introduction or conclusion.

Include no pre-amble for either section.

Target around 100 words, crisply previewing (for introduction) or recapping (for conclusion) all of the sections of the report.

Use markdown formatting. 

For your introduction, create a compelling title and use the # header for the title.

For your introduction, use ## Introduction as the section header. 

For your conclusion, use ## Conclusion as the section header.

Here are the sections to reflect on for writing: {formatted_str_sections}"""




#takes state info, combines with instructions + writes a report
def write_report(state: ResearchGraphState):
    
    #state keys
    sections = state['sections']
    topic = state['topic']

    #concat all sections
    formatted_str_sections = "\n\n".join([f"{section}" for section in sections])

    #summarize
    system_message = report_writer_instructions.format(topic=topic, context=formatted_str_sections)
    report = llm.invoke([SystemMessage(content=system_message)] + [HumanMessage(content=f'Write a report based on these memos.')])
    
    #return to state
    return {'content': report.content}



#takes state info, combines with instructions + writes the introduction
def write_introduction(state: ResearchGraphState):
    
    #state keys
    sections = state["sections"]
    topic = state["topic"]

    #concat all sections together
    formatted_str_sections = "\n\n".join([f"{section}" for section in sections])
    
    #summarize the sections into a final report
    instructions = intro_conclusion_instructions.format(topic=topic, formatted_str_sections=formatted_str_sections)    
    intro = llm.invoke([instructions]+[HumanMessage(content=f"Write the report introduction")]) 

    #return to state
    return {"introduction": intro.content}



#takes state info, combines with instructions + writes the conclusion
def write_conclusion(state: ResearchGraphState):
    
    #state keys
    sections = state["sections"]
    topic = state["topic"]

    #concat all sections together
    formatted_str_sections = "\n\n".join([f"{section}" for section in sections])
    
    #summarize the sections into a final report
    instructions = intro_conclusion_instructions.format(topic=topic, formatted_str_sections=formatted_str_sections)    
    conclusion = llm.invoke([instructions]+[HumanMessage(content=f"Write the report conclusion")]) 

    #return to state
    return {"conclusion": conclusion.content}



#creates the final report with all components
def finalize_report(state: ResearchGraphState):

    #save full final report
    content = state["content"]

    #criteria to create
    if content.startswith("## Insights"):
        content = content.strip("## Insights")
    if "## Sources" in content:
        try:
            content, sources = content.split("\n## Sources\n")
        except:
            sources = None
    else:
        sources = None

    #structure
    final_report = state["introduction"] + "\n\n---\n\n" + content + "\n\n---\n\n" + state["conclusion"]

    #add sources
    if sources is not None:
        final_report += "\n\n## Sources\n" + sources

    #final report to state
    return {"final_report": final_report}