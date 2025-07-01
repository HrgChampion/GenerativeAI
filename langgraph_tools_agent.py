from IPython.display import Image,display
from langgraph.graph import StateGraph,START
from langchain_openai import ChatOpenAI
import requests
from langchain_core.messages import SystemMessage,HumanMessage
from langgraph.graph import MessagesState
from langchain_community.tools import DuckDuckGoSearchRun
from langgraph.prebuilt import ToolNode,tools_condition
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages
from dotenv import load_dotenv
from langgraph.prebuilt import ToolNode,tools_condition

load_dotenv()


class State(TypedDict):
      messages:Annotated[list,add_messages]

def search_duckduckgo(query:str):
      """Searches DuckDuckGo using LangChain's DuckDuckGoSearchRun tool."""
      search = DuckDuckGoSearchRun()
      return search.invoke(query)

result = search_duckduckgo("Delhi's wetaher today")
print(result)

def multiply(a:int,b:int) -> int:
      """Multiplies two integers."""
      return a*b

def add(a:int,b:int):
    """Adds two integers."""
    return a+b

# from langchain_openai import ChatOpenAI
# llm = ChatOpenAI(temperature=0,api_key="api key",model="gpt-4o-mini")

from langchain_google_genai import ChatGoogleGenerativeAI
import os

os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

# Step 2: Create Gemini LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash-latest",
    temperature=0.2,
    max_output_tokens=512,
)
llm.invoke("hello").content

tools=[search_duckduckgo,add,multiply]
llm_with_tools=llm.bind_tools(tools)

def chatbot(state:State):
    return {"messages":[llm_with_tools.invoke(state["messages"])]}


graph_builder = StateGraph(State)

graph_builder.add_node("assistant",chatbot)
graph_builder.add_node("tools",ToolNode(tools))

graph_builder.add_edge(START,"assistant")
graph_builder.add_conditional_edges("assistant",tools_condition)
graph_builder.add_edge("tools","assistant")
react_graph= graph_builder.compile()

display(Image(react_graph.get_graph().draw_mermaid_png()))

response = react_graph.invoke({"messages":[HumanMessage(content="Multiply 10 by 2 and add 5.What is the delhi's weather from the web search.")]})
print(response["messages"])

for m in response["messages"]:
    m.pretty_print()