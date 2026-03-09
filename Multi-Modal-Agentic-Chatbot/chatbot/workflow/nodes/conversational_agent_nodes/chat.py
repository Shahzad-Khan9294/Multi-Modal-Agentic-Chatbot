import uuid
from langchain_core.runnables import RunnableConfig
from ...states import GraphState
from ...utils import get_langchain_vllm_model
from langchain_core.messages import SystemMessage,  HumanMessage, RemoveMessage
import time

async def chat(state: GraphState, config: RunnableConfig) -> dict:
    # user_msg = state["question"]
    # state["messages"].append({"role": "human", "content": user_msg})
    # print("\nentering simple_chat with user message:", user_msg, "\n")
    start_time = time.perf_counter()
    prompt = """
    You are a helpful TAG AI conversational assistant. Answer the user’s question clearly and concisely.

    Instructions:

    Provide direct, simple explanations.
    If the question is instructional, give step-by-step guidance.
    If the question is definitional, give a clear definition and short example.
    If translation is requested, translate accurately and naturally.
    Keep responses concise and user-friendly (avoid unnecessary detail).
    Always use chat history to answer the question. If the question is not related to the chat history,
    say that you don't know the answer.
    """
    # trace = config["configurable"].get("trace", {}).copy()
    # trace.update({'span_id': str(uuid.uuid4()), 'span_name': 'Simple Chat'})

    message=HumanMessage(content=state['question'])
    messages = [SystemMessage(content=prompt)] + state["messages"]
    llm = get_langchain_vllm_model(model="unsloth/gemma-3-12b-it-bnb-4bit")
    response = await llm.ainvoke(messages)


    print("Messages in state:", len(state["messages"]))
    # keep last 6 messages from state messages and remove other messages
    if len(state["messages"]) > 6:
        keep_messages = state["messages"][-6:].copy()
        print("keep_messages:", len(keep_messages))
        state["messages"].clear()
        state["messages"] = keep_messages
        print("Messages in state after deletion:", len(state["messages"]))

    end_time = time.perf_counter()
    inference_time = end_time - start_time
    print(f"Chat node inference time: {inference_time:.4f} seconds")
    return {
        "messages": [message, response],
    }