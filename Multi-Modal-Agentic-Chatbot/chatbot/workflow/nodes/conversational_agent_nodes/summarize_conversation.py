from ...states import GraphState
from ...utils import get_langchain_vllm_model
from langchain_core.messages import (
    HumanMessage,
    RemoveMessage,
)
from langchain_core.runnables import RunnableConfig
import time

def summarize_conversation(state: GraphState, RunnableConfig) -> dict:

    start_time = time.perf_counter()
    # First, we get any existing summary
    summary = state.get("summary", "")

    # Create our summarization prompt
    if summary:

        # A summary already exists
        summary_message = (
            f"This is a summary of the conversation to date: {summary}\n\n"
            "Extend the summary by taking into account the new messages above:"
        )

    else:
        summary_message = "Create a summary of the conversation above:"

    llm = get_langchain_vllm_model()
    # Add prompt to our history
    
    messages = state["messages"] + [HumanMessage(content=summary_message)]
    response = llm.invoke(messages)

    # Delete all but the 2 most recent messages
    delete_messages = [RemoveMessage(id=m.id) for m in state["messages"][:-2]]
    # return {"summary": response.content, "messages": delete_messages}

    end_time = time.perf_counter()
    inference_time = end_time - start_time
    print(f"---------------------- Summarize Conversation inference time ----------------------: {inference_time:.4f} seconds")
    return {"summary": response.content, "messages": delete_messages}
