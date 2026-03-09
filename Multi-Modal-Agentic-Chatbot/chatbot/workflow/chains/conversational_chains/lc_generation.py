from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts.chat import HumanMessagePromptTemplate
from ...utils import get_langchain_vllm_model
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate


def get_lc_generation_chain(model):    
    llm = get_langchain_vllm_model(model=model)
    template = """Your role is to act as a real estate investment advisor answering the user's request as possible.
        You are provided with a text made up of several small messages, all related to the user's request.
        From these messages, you are tasked to formulate a cohesive, engaging response addressed to the user's query.
        Your response should be in the English language as the user's request. It is vital that you utilise 
        only the provided text, refrain from creating anything from imagination or using other information 
        sources when composing a response. The response should be concise, and it should correspond maximally 
        to the user's request. Here is the text: ''' {context} '''. Here is the user's request: {question}"""
    prompt = ChatPromptTemplate(input_variables=['context', 'question'],
                                messages=[
                                    HumanMessagePromptTemplate(
                                    prompt=PromptTemplate(
                                    input_variables=['context', 'question'],
                                    input_types={}, partial_variables={},
                                    template=template),
                                    additional_kwargs={})])

    lc_generation_chain = prompt | llm | StrOutputParser()
    return lc_generation_chain