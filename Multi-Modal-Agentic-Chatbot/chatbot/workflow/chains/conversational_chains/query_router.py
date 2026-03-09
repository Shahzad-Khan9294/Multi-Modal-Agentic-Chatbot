from typing import Optional
from typing import Literal, List
from pydantic import BaseModel, Field
from ...utils import get_langchain_vllm_model_sr
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser
from datetime import datetime
from typing import Literal


class RouteQuery(BaseModel):
    """Route a user query to the most relevant datasource and extract document names from the user questions."""
 
    route: Literal["RAG", "LongContext" , "ResourceList", "CHAT", "Risk"] = Field(
        ...,
        description="The route to take for the user question. RAG for specific external context, LongContext for external long context or multiple documents," \
        " CHAT for general questions that do not text retrieval, Risk for asking for score of any risk category name.",)
    document_names: List[str] = Field(description="extract name or names of the documents or  article mention in the user questions")
    heading: List[str] = Field(description="extract the heading of the document or article from the user questions")
    resource_type_list : List[str] = Field(description="""contains any resource type like "Analysis" if there is analysis/articles mentioned in user question', return "Document" if there is document/s mentioned in user question',
    return "Calendar" if there is events/calendars mentioned in user question, return "Stakeholder" if there is stakeholders mentioned in user question""")
    detail_info : str = Field("response 'yes' if the user question is demanding further details, expansion, or additional information about some topic or content. If not then respond as 'no'")
    risk_category_names: List[str] = Field(description="Extract full risk-related phrases as expressed in the user question. If a geographic or organizational entity is tied to a risk/topic, return the combined phrase as a single risk category name.")
    my_ci_score: str = Field("If the user asks about a score and uses the word **“my”** to mean the score belongs to them (for example, *“my CI score”* or *“my risk category score”*), return 'yes', If they are not talking about their own score, return 'no'.")
    
def get_question_router(model, max_tokens=1000):
    llm = get_langchain_vllm_model_sr(model=model, max_tokens=max_tokens)
    system = f"""
    Act as an expert routing agent. Your task is to analyze a user question and decide the most relevant route for it and
    also return the information related to document_names or headings or detail_info or resource_types_list or risk_category_names or my_ci_score present in the user question..

    **STRICT RULES**:
    - Return ONLY **valid JSON**
    - Do NOT use bullet points
    - Do NOT wrap output in markdown
    - Use empty lists instead of null 

    **RESOURCE TYPES**:
    - Stakeholder → people, organizations, companies, actors
    - Analysis → articles, analysis, reports, research
    - Document → documents, files, PDFs, papers
    - Event → calendar, events, meetings, event titles
 
    **YOUR OUTPUT MUST INCLUDE**:
    - route: one of "RAG", "LongContext", "ResourceList", "Risk" or "CHAT"  
    - document_names: a list of any document names mentioned in the question (can be empty)
    - heading: a list of any heading of the document mentioned in the question (can be empty)
    - detail_info: Set `detail_info` to 'yes' only if the question explicitly references prior content or includes clear follow-up terms (e.g., "more", "further", "additional", "expand", "also", "what about", "from above"); otherwise set it to 'no' for standalone or new questions, and never infer intent from topic similarity.
    - resource_type_list: extract resource type/s: Stakeholder or Analysis or Document or Event and
      keeping the first letter capital like ("Stakeholder" in case of stakeholder/s is mentioned or "Analysis" in case of analysis/articles mentioned 
      or "Document" in case of document/s mentioned or "Event" in case of events/calenders mentioned in the user question)
    - risk_category_names: a list of full risk-related phrases as expressed in the user question. If a geographic or organizational entity is tied to a risk/topic, return the combined phrase as a single risk category name (can be empty)
    - my_ci_score: if user question mentions a word *my* that directly modifies or refers to a score (e.g., "my ci score", "my political economy score") return 'yes', otherwise return 'no'.

    **ROUTING RULES**:
    1. CHAT
    - Use only when the question is clearly conversational, instructional, or unrelated to the knowledge base.  
    - These are general inquiries that do not depend on retrieving information from documents.  
 
    2. RAG
    - Use when the question depends on facts, data, descriptions, or context from external documents.  
    - This includes questions asking about concepts, behaviors, motivations, statistics, definitions, or any content that is expected to come from the documents.
    - If the query can reasonably be answered by retrieving relevant passages, always use RAG.
 
    3. LongContext
    - Use only when the user explicitly requests a summary, comparison, or synthesis of multiple long documents or extended passages.  
    - Do not assign to LongContext unless the user question clearly mentioning for summarization across multiple documents or articles/analysis.
    - If user question ask for summary and mention any proper document or article name, use LongContext always.
 
    4. ResourceList
    - Use only when the user explicitly requests for a list of any resource type word like: stakeholder/article/calendar/document or more than one resource type words.
    - If the word 'list' or 'List' is present and question clearly mentioning any resource type word like: "stakeholder" or "article" or "calendar" or "document" or more than one resource type words use ResourceList.
    - read the whole user question fully and extract resource type/s like "stakeholder" or "analysis or article" or "event or calendar" or "document" mentioned in the question and list them in `resource_type_list` in format:
      keeping the first letter capital return "Analysis" if there is analysis/articles mentioned in user question', return "Document" if there is document/s mentioned in user question',
      return "Calendar" if there is events/calendars mentioned in user question, return "Stakeholder" if there is stakeholders mentioned in user question.

    5. Risk
    - Use only when ask for **score/ci score of any risk category name/s** is mentioned in the user question.
    - If a date range is mentioned exactly or relative (e.g, last/previous 10 days", "last/previous month", "last/previous week", "yesterday")

    **VERY IMPORTANT**:
    - If the question contains multiple resource nouns (for example: "stakeholders and articles and documents"),
    then resource_type_list MUST include ALL of them.
    - Never collapse multiple types into one.
    - Always include every resource type that appears anywhere in the question.

    **For risk_category_names ONLY**:
    - If an entity (country, organization, region) and a risk/topic appear contiguously or are logically bound in the question, they MUST be returned as ONE combined phrase.
    - DO NOT separate entity names into other fields or split them across outputs.

    **Guidelines**:
    - Always choose the most specific route that applies.  
    - If a question could fit multiple routes, prioritize in this order: **RAG > LongContext > ResourceList > CHAT**.  
    - Assume that content-related questions should be routed to RAG unless there is an explicit request for summarization.  
    - Extract any document names mentioned in the question and list them in `document_names`.
    - Set `detail_info` to 'yes' ONLY if the user’s question explicitly refers to prior content or contains clear follow-up indicators (e.g., "more", "further", "additional", "expand", "elaborate", "also", "continue", "what about", "based on that", "from above", "earlier", "previously mentioned") 
      or asks to refine or add details to already stated information; otherwise set `detail_info` to 'no' for standalone questions, topic restatements, or new questions on the same subject, and do NOT infer follow-up intent from topic similarity—when in doubt, default to 'no'.
    - Extract resource type/s and list them in `resource_type_list`, in user question there can be more than one resource types mentioned.
    - Extract any resource type word/s like "stakeholder" or "analysis" or "article" or "event" or "calendar" or "document" mentioned in the question and list them in `resource_type_list`,
      keeping the first letter capital return "Analysis" if there is analysis/articles mentioned in user question', return "Document" if there is document/s mentioned in user question',
      return "Calendar" if there is events/calendars mentioned in user question, return "Stakeholder" if there is stakeholders mentioned in user question.   
    - Extract a small set of unique, meaningful keywords or multi-word phrases from the USER QUESTION that best capture its overall intent. **Preserve the longest meaningful semantic span**: if removing any word changes real-world meaning or breaks the link between an entity and its risk/topic, the phrase MUST remain intact.
      Treat entity + domain/topic combinations (e.g., country + politics/economy/risk/conflict/regulation/security) as ONE indivisible keyphrase and DO NOT split them into sub-phrases. Preserve domain-specific phrases as single units whenever they convey a coherent concept. Ignore filler words and stopwords without losing semantic meaning.
      Only generalize wording if it does not break a preserved multi-word phrase; never generalize by splitting phrases. Return only a list of strings under `risk_category_names`, with no extra text or formatting. Apply the same rules for both RAG and Risk-type queries.
    - If the user question explicitly refers to a score as their own using first-person ownership (e.g., phrases like "my ci score" or "my [risk category] score") set it as 'yes', and if user question does not mentions clear ownership using *my*, set it to 'no'. Respond strictly with "yes" or "no" and return response in `my_ci_score`.
    """
    parser = PydanticOutputParser(pydantic_object=RouteQuery)
    route_prompt = ChatPromptTemplate(
        [
            ("system", system),
            ("human", "User question: \n\n {question}, Output: {format_instructions} "),
        ],
        partial_variables={"format_instructions": parser.get_format_instructions()}
    )
    question_router = route_prompt | llm | parser 
    return question_router






    # - If the user question (e.g, how much score of xyz_thing of been changed in last/previous month till toady?) mention or ask about some relevent date range/s but that date range is becoming the date range from previous year by calculating it with today's date: {today_date},
    #   then set that date range accordingly with respect to previous year, month and day.
    # - If the user question is like (e.g, how much score of xyz_thing of last/previous month been changed?), then you have to set the start date as the very first date range of that month and 
    #   end date as the **last date of that same month(mentioned in user question) at which it finishes**.
    # - if the user question just singly sepcifies any month name but does not mention or hints about of which year, then check today's date and month : {today_date} and
    #   check whether that single specified month(in the user question) is that same month or any previous month before today's date, if it is a month before todays date then set the start date as the starting date of that month and the end date as the finish date of that month and kepp the year same as
    #   in the todays date mentioned and if the month is after the todays date then set its start date as the start date of that month and end date its finish date of that month but keep the year as the previous year of current todats date year.
    # - If a date range is mentioned, return start_date and end_date objects, set the start date **with the date range mentioned in the user question** and the end date with the todays date: {today_date} and set both objects in `date_range`. if there is no date range mentioned in the user question then keep `date_range` empty.

 