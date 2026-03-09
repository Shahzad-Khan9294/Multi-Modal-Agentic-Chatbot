import asyncio
import operator
from typing import List
from langgraph.constants import Send
from langchain_core.documents import Document
from langchain.chains.combine_documents.reduce import (
    acollapse_docs, split_list_of_docs,)
# from .workflow.states import GraphState
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import AIMessage, HumanMessage
from typing import Literal
import os
from .states import GraphState
from langchain_openai import ChatOpenAI


LLM_VLLM_API_BASE = os.getenv("LLM_VLLM_API_BASE")
LLM_VLLM_API_KEY = os.getenv("LLM_VLLM_API_KEY")
Model = os.getenv("Model")

def get_langchain_vllm_model(Model):

    llm = ChatOpenAI(
        model=Model,
        openai_api_key=LLM_VLLM_API_KEY,
        openai_api_base=LLM_VLLM_API_BASE,
        temperature=0.1,
        # max_tokens = 10000,
    )
    return llm


map_template = """Write a concise summary of the following: {context}
coving all major aspects from the provoded context"""
reduce_template = """
    You are an expert summarizer. Your task is to generate a consolidated summary
    in a clear, readable form.
    Analyze all of them carefully and generate a **final, consolidated summary** that:
        - Covers **all key points and themes** mentioned in the summaries.
        - Integrates overlapping ideas without repetition.
        - Ensures no major aspect from any summary is omitted.
        - Presents the content clearly and cohesively.
    ##Summaries:
        {docs}
    Take these Summaries and distill it into a final, consolidated summary
    of the main themes."""

llm = get_langchain_vllm_model(Model)
map_prompt = ChatPromptTemplate([("human", map_template)])
reduce_prompt = ChatPromptTemplate([("human", reduce_template)])
map_chain = map_prompt | llm | StrOutputParser()
reduce_chain = reduce_prompt | llm | StrOutputParser()
 
 
token_max = 10000
def length_function(documents: List[Document]) -> int:
    total_tokens = sum(llm.get_num_tokens(doc.page_content) for doc in documents)
    print("total tokens: ", total_tokens)
    return total_tokens

async def map_and_generate_summary(state: GraphState):
    if not state['documents']:
        print("No documents to summarize.")
        return {"summaries": []}
    state["summaries"] = await map_chain.abatch(inputs=[{"context": doc.page_content} for doc in state['documents']])
    print(f"Total summaries generated: {len(state['summaries'])}")
    return {"summaries": state['summaries']}
 
def collect_summaries(state: GraphState):
    state["collapsed_summaries"] = [Document(summary) for summary in state.get("summaries", [])]
    print("Collect summaries#:", len(state["collapsed_summaries"]))
    return {"collapsed_summaries": state["collapsed_summaries"]}
 
async def collapse_summaries(state: GraphState):
    doc_lists = split_list_of_docs(state["collapsed_summaries"], length_function, token_max)
    print("doc_lists", len(doc_lists))
    results = []
    for doc_list in doc_lists:
        print()
        print("collapsing len", len(doc_list))
        print("--------------------------- collapsing ------------------------------------ ")
        # print(doc_list)
        results.append(await acollapse_docs(doc_list, reduce_chain.ainvoke))
    print("results", len(results))
    state["collapsed_summaries"] = results
    return {"collapsed_summaries": results}
 

def should_collapse(
    state: GraphState,
) -> Literal["collapse_summaries", "generate_final_summary"]:
    num_tokens = length_function(state["collapsed_summaries"])

    print("Number of tokens in collapsed summaries:", num_tokens)
    if num_tokens > token_max and len(state["collapsed_summaries"]) > 1:
        return "collapse_summaries"
    else:
        return "generate_final_summary"
 
async def generate_final_summary(state: GraphState):
    print("Generating final summary...")
    # print("collapsed_summaries", state["collapsed_summaries"])
    input_data = {"input": {"docs": state["collapsed_summaries"]}}
    response = await reduce_chain.ainvoke(**input_data)
    # print("Final summary generated:", response)
    return {"final_summary": response}
 
 
# async def run_summary_flow(chunks_data: List[str]):
#     await map_and_generate_summary(chunks_data)
#     collect_summaries()
#     while True:
#         next_node = should_collapse()
#         print(next_node)
#         if next_node == "collapse_summaries":
#             await collapse_summaries()
#         else:
#             final_result = await generate_final_summary()
#             final_summary = final_result["messages"][0] if final_result.get("messages") else None
#             break
#     print("Summary workflow complete!")
#     return final_summary