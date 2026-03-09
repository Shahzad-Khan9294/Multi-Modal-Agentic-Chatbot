from ...utils import get_langchain_vllm_model
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.prompts.chat import HumanMessagePromptTemplate, SystemMessagePromptTemplate

def get_adjacent_generation_chain(model):  

    llm = get_langchain_vllm_model(model=model)
    template = """
    Act as an expert response-generation agent handling FOLLOW-UP and
    "DEMANDING MORE INFORMATION" queries (e.g., "tell me more", "explain further", 
    "give more details about that").

    Your primary objective is to extend the discussion WITHOUT repeating information
    that has already been covered in the recent conversation → (CHAT HISTORY).
    **DO NOT USE** or add the chat history response content in the final response → (inspite add new information/content from the provided context)

    ##CORE OBJECTIVE:
    Generate a response that:
        - Relevant to the USER QUESTION
        - Use the context as the main source for new information.
        - Adds ONLY new information → (Not including previous (CHAT HISTORY) information) 
        - Avoid using information/content already present in the CHAT HISTORY
        
    ##STEP-BY-STEP INSTRUCTIONS:
        1) Examine the CONTEXT and extract:
            - Information that is relevant to the USER QUESTION
            - Information that has NOT appeared in the CHAT HISTOR

        2) Generate the final response by:
            - Using ONLY new, non-repeated information from the CONTEXT
            - Expanding the topic with deeper explanations, examples, or breakdowns
            - Completely excluding any content already covered in CHAT HISTORY

    ##STRICT RULES:
        - **DO NOT USE** or add the chat history response content in the final response (inspite add new information/content from the provided context)
        - Do NOT introduce any information not explicitly present in the CONTEXT.
        - Do NOT rely on general knowledge, assumptions, or model training data.
        - If the CONTEXT does not contain new information beyond what was already discussed,
        clearly state that no additional details are available.

    ##ANSWER STYLE:
        - Descriptive and in-depth (appropriate for "tell me more" queries)
        - Use clear headings and sub-headings
        - Stay directly relevant to the USER QUESTION
        - Match the language and tone of the user's input
        - Be concise but informative

    ##FAIL-SAFE BEHAVIOR:
    If no NEW relevant information exists in the CONTEXT beyond what is already
    covered in CHAT HISTORY, respond with:
    "There is no additional information available in the provided context beyond what has already been discussed!"
    """

    question_template = """ ### Provided Context: ''' {context} ''' \n\n ### User Question: {question}  \n\n ### Chat History: ''' {chat_history} ''' """
    prompt = ChatPromptTemplate(input_variables=['context', 'question', 'chat_history'],
                                messages=[
                                    SystemMessagePromptTemplate(
                                    prompt=PromptTemplate(
                                    template=template),
                                    additional_kwargs={}),
                                    HumanMessagePromptTemplate(
                                    prompt=PromptTemplate(
                                    input_variables=['context', 'question', 'chat_history'],
                                    input_types={}, partial_variables={},
                                    template=question_template),
                                    additional_kwargs={})])

    adjacent_generation_chain = prompt | llm | StrOutputParser()
    return adjacent_generation_chain
