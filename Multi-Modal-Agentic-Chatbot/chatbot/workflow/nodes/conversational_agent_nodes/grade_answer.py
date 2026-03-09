from langchain_core.runnables import RunnableConfig
from ...states import GraphState
from ...chains import get_answer_grader
import time

async def grade_answer(state: GraphState, config: RunnableConfig) -> dict:
    """
    Grades the answer based on whether it resolves the question.
    """
    start_time = time.perf_counter()
    print("---GRADE ANSWER---")
    
    question = state["question"]
    generation = state["messages"][-1].content  # Assuming the last message is the LLM generation
    
    # trace = config["configurable"].get("trace", {}).copy()
    # trace.update({'span_id': 'grade_answer', 'span_name': 'Grade Answer'})
    
    answer_grader = get_answer_grader(model="unsloth/gemma-3-12b-it-bnb-4bit")
    print("answer_grader",answer_grader)
    
    graded_answer = await answer_grader.ainvoke({"question": question, "generation": generation})
    
    end_time = time.perf_counter()
    inference_time = end_time - start_time
    print(f"---------------------- Generate inference time ----------------------: {inference_time:.4f} seconds")
    return {"answer_validation": graded_answer.binary_score}