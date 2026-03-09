from typing import Dict, Any, List
from ...chains.conversational_chains.divide_query import get_divide_chain
import time



async def divide_query(state, config) -> Dict[str, Any]:
    """
    Run the divide_query chain to split a clarified or original question into smaller queries.
    Returns a dict containing divided queries and updates the state.
    """
    start_time = time.perf_counter()

    question = (state.get("question") or "").strip()
    if not question:
        print("⚠️ No question found in state for divide_query.")
        return {"generated_questions": []}

    divide_chain = get_divide_chain(
        model="unsloth/gemma-3-12b-it-bnb-4bit"
    )

    if hasattr(divide_chain, "ainvoke"):
        divide_result = await divide_chain.ainvoke({"question": question})
    else:
        loop = asyncio.get_running_loop()
        divide_result = await loop.run_in_executor(
            None, lambda: divide_chain.invoke({"question": question})
        )

    print("---DIVIDE QUERY RESULT---")
    print(divide_result)

    queries = divide_result.queries
    
    # Ensure it's always a list
    if not isinstance(queries, list):
        queries = [str(queries)]

    print("---DIVIDED QUERIES---")
    for i, q in enumerate(queries, start=1):
        print(f"Query {i}: {q}")

    # Update state
    state["generated_questions"] = queries

    end_time = time.perf_counter()
    inference_time = end_time - start_time
    print(f"---------------------- Divide Query inference time ----------------------: {inference_time:.4f} seconds")

    return {"generated_questions": queries}

    