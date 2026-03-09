import operator
from langgraph.constants import Send
from typing import Annotated, List, Literal
from langchain_core.documents import Document

from langchain.chains.combine_documents.reduce import (
    acollapse_docs,
    split_list_of_docs,
)
from ...states import GraphState
from langchain_core.messages import AIMessage, HumanMessage


def length_function(documents: List[Document]) -> int:
    """Get number of tokens for input contents."""
    print("total tokens: ",sum(llm.get_num_tokens(doc.page_content) for doc in documents))
    return sum(llm.get_num_tokens(doc.page_content) for doc in documents)

token_max = 80000

#----------------------------------- chains

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from ...utils import get_langchain_vllm_model

map_template = "Write a concise summary of the following: {context}."

reduce_template = """
The following is a set of summaries and user question:
Question:
{question}

Summaries:
{docs}
Take these and distill it into a final, consolidated summary
of the main themes.
"""

llm = get_langchain_vllm_model(model="unsloth/gemma-3-12b-it-bnb-4bit")
map_prompt = ChatPromptTemplate([("human", map_template)])
reduce_prompt = ChatPromptTemplate([("human", reduce_template)])

map_chain = map_prompt | llm | StrOutputParser()
reduce_chain = reduce_prompt | llm | StrOutputParser()

#----------------------------------- Nodes

# Here we generate a summary, given a document
async def generate_summary(state: GraphState):
    response = await map_chain.ainvoke(state["content"])
    return {"summaries": [response]}


# Here we define the logic to map out over the documents
# We will use this an edge in the graph
def map_summaries(state: GraphState):
    print("Mapping summaries over documents text :", len(state["documents"]))
    # We will return a list of `Send` objects
    # Each `Send` object consists of the name of a node in the graph
    # as well as the state to send to that node
    if not state["documents"]:
        print("No documents to summarize.")
        return  [Send("generate_summary", {"content": "No content provided."})]
    return [
        Send("generate_summary", {"content": content}) for content in state["documents"]
    ]


# Add node to store summaries for collapsing
def collect_summaries(state: GraphState):
    # print("Collecting summaries:", state["summaries"])
    return {
        "collapsed_summaries": [Document(summary) for summary in state["summaries"]]
    }


# Modify final summary to read off collapsed summaries
async def generate_final_summary(state: GraphState):
    input_data = {"input":{"question":state["question"], "docs":state["collapsed_summaries"]}}
    response = await reduce_chain.ainvoke(**input_data)
    print("Final summary generated:", response)
    state["messages"].append(HumanMessage(content=state["question"]))

    response = AIMessage(content=response)

    return {"messages": [response]}


# Add node to collapse summaries
async def collapse_summaries(state: GraphState):

    # if length_function(state["collapsed_summaries"]) <= token_max:
    #     print("No need to collapse, returning current summaries")
    #     doc_lists.append(state["collapsed_summaries"])
    # else:
    doc_lists = split_list_of_docs(
        state["collapsed_summaries"], length_function, token_max
    )
    print("doc_lists",len(doc_lists))  # Debugging: Check how many chunks we have
    results = []

    for doc_list in doc_lists:
        print()
        print("collapsing len", len(doc_list))
        print("--------------------------- collapsing ------------------------------------ ")
        print(doc_list)
        results.append(await acollapse_docs(doc_list, reduce_chain.ainvoke))
    print("results",len(results))
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
