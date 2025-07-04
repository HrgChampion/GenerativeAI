from typing import List,Dict
from langgraph.graph import StateGraph,START,END
from langchain_ollama.llms import OllamaLLM

# Step 1: Define State
class State(Dict):
    messages:List[Dict[str,str]]

# Step 2: Initialize the StateGraph
graph_builder = StateGraph(State)


# Step 3 : Initialize the LLM
llm = OllamaLLM(model="llama3.1")

# Step 4 : Define Chatbot function
def chatbot(state:State):
    response = llm.invoke(state["messages"])
    state["messages"].append({"role": "assistant", "content": response})
    return {"messages": state["messages"]}

# Add nodes and edges
graph_builder.add_node("chatbot",chatbot)
graph_builder.add_edge(START, "chatbot")
graph_builder.add_edge("chatbot", END)

#Compile the graph
graph = graph_builder.compile()

#Stream Updates
def stream_graph_updates(user_input:str):
    # Initialize the state with user input
    state = {"messages": [{"role": "user", "content": user_input}]}
    for event in graph.stream(state):
        for value in event.values():
            # Print the assistant's response
            print("Assistant:",value["messages"][-1]["content"])

# Run the chatbot in a loop
if __name__ == "__main__":
    while True:
        try:
            user_input = input("User: ")
            if user_input.lower() in ["exit", "quit","q"]:
                print("Goodbye!")
                break
            stream_graph_updates(user_input)
        except Exception as e:
            print(f"An error occurred: {e}")
            break


