from ...chains import get_lc_generation_chain
from ...states import GraphState
from langchain_core.runnables import RunnableConfig
from langchain_core.messages import AIMessage
import uuid
import time

async def lc_generate(state: GraphState, config: RunnableConfig):
    """
    Generates a response using the LangChain generation chain.
    """
    start_time = time.perf_counter()
    print("---LC GENERATE---")
    
    question = state["question"]
    documents = state["documents"]
    
    # trace = config["configurable"].get("trace", {}).copy()
    # trace.update({'span_id':str(uuid.uuid4()) , 'span_name': 'Long Context Generate'})

    lc_generation_chain = get_lc_generation_chain(model="unsloth/gemma-3-12b-it-bnb-4bit")
    response = await lc_generation_chain.ainvoke({"context": documents, "question": question})
    response = AIMessage(content=response)

    print("Response from LC generation chain:", response)

    end_time = time.perf_counter()
    inference_time = end_time - start_time
    print(f"---------------------- LC Generate inference time ----------------------: {inference_time:.4f} seconds")
    return {"messages": [response]}