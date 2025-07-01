
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage

api_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=300)
wiki_tool = WikipediaQueryRun(api_wrapper=api_wrapper)
     

wiki_tool.run({"query": "AI agents"})
     
# 'Page: Intelligent agent\nSummary: In artificial intelligence, an intelligent agent is an entity that perceives its environment, takes actions autonomously to achieve goals, and may improve its performance through machine learning or by acquiring knowledge. Leading AI textbooks define artificial intel'

llm = ChatOpenAI(temperature=0, api_key="sk*******************M7", model="gpt-4o-mini")

     

tools = [wiki_tool]

# Tool binding
llm_with_tools = llm.bind_tools(tools)

#Tool calling
result = llm_with_tools.invoke("Hello world!")
result.content
     
# 'Hello! How can I assist you today?'



agent_executor = create_react_agent(llm, tools)
     



#First up, let's see how it responds when there's no need to call a tool:
response = agent_executor.invoke({"messages": [HumanMessage(content="hi!")]})

response["messages"]
     
# [HumanMessage(content='hi!', additional_kwargs={}, response_metadata={}, id='6945f09a-4798-49c1-8d4a-e369f165b671'),
#  AIMessage(content='Hello! How can I assist you today?', additional_kwargs={'refusal': None}, response_metadata={'token_usage': {'completion_tokens': 11, 'prompt_tokens': 83, 'total_tokens': 94, 'completion_tokens_details': {'accepted_prediction_tokens': 0, 'audio_tokens': 0, 'reasoning_tokens': 0, 'rejected_prediction_tokens': 0}, 'prompt_tokens_details': {'audio_tokens': 0, 'cached_tokens': 0}}, 'model_name': 'gpt-4o-mini-2024-07-18', 'system_fingerprint': 'fp_72ed7ab54c', 'finish_reason': 'stop', 'logprobs': None}, id='run-15409985-a239-4005-a880-1bbdff55e74a-0', usage_metadata={'input_tokens': 83, 'output_tokens': 11, 'total_tokens': 94, 'input_token_details': {'audio': 0, 'cache_read': 0}, 'output_token_details': {'audio': 0, 'reasoning': 0}})]

print(response["messages"][-1].content)
     
# Hello! How can I assist you today?


#First up, let's see how it responds when there's no need to call a tool:
response = agent_executor.invoke({"messages": [HumanMessage(content="what is agentic ai")]})

response["messages"]
     
# [HumanMessage(content='what is agentic ai', additional_kwargs={}, response_metadata={}, id='cc704242-a6b6-4beb-8540-1189d8ee727d'),
#  AIMessage(content='', additional_kwargs={'tool_calls': [{'id': 'call_nJETD0JhkN3atAANJQ5Jh0zs', 'function': {'arguments': '{"query":"Agentic AI"}', 'name': 'wikipedia'}, 'type': 'function'}], 'refusal': None}, response_metadata={'token_usage': {'completion_tokens': 16, 'prompt_tokens': 86, 'total_tokens': 102, 'completion_tokens_details': {'accepted_prediction_tokens': 0, 'audio_tokens': 0, 'reasoning_tokens': 0, 'rejected_prediction_tokens': 0}, 'prompt_tokens_details': {'audio_tokens': 0, 'cached_tokens': 0}}, 'model_name': 'gpt-4o-mini-2024-07-18', 'system_fingerprint': 'fp_bd83329f63', 'finish_reason': 'tool_calls', 'logprobs': None}, id='run-72ada718-03e2-4ff9-ad0b-f1c9013bbc59-0', tool_calls=[{'name': 'wikipedia', 'args': {'query': 'Agentic AI'}, 'id': 'call_nJETD0JhkN3atAANJQ5Jh0zs', 'type': 'tool_call'}], usage_metadata={'input_tokens': 86, 'output_tokens': 16, 'total_tokens': 102, 'input_token_details': {'audio': 0, 'cache_read': 0}, 'output_token_details': {'audio': 0, 'reasoning': 0}}),
#  ToolMessage(content='Page: Intelligent agent\nSummary: In artificial intelligence, an intelligent agent is an entity that perceives its environment, takes actions autonomously to achieve goals, and may improve its performance through machine learning or by acquiring knowledge. Leading AI textbooks define artificial intel', name='wikipedia', id='8836c552-b430-4d34-a3e1-627a4ef420fe', tool_call_id='call_nJETD0JhkN3atAANJQ5Jh0zs'),
#  AIMessage(content='Agentic AI refers to intelligent agents in artificial intelligence that can perceive their environment, take autonomous actions to achieve specific goals, and potentially improve their performance through learning or acquiring knowledge. These agents operate independently and can adapt their behavior based on their experiences.', additional_kwargs={'refusal': None}, response_metadata={'token_usage': {'completion_tokens': 51, 'prompt_tokens': 159, 'total_tokens': 210, 'completion_tokens_details': {'accepted_prediction_tokens': 0, 'audio_tokens': 0, 'reasoning_tokens': 0, 'rejected_prediction_tokens': 0}, 'prompt_tokens_details': {'audio_tokens': 0, 'cached_tokens': 0}}, 'model_name': 'gpt-4o-mini-2024-07-18', 'system_fingerprint': 'fp_72ed7ab54c', 'finish_reason': 'stop', 'logprobs': None}, id='run-a8460c47-2361-4df5-8882-3f46a1c2dde2-0', usage_metadata={'input_tokens': 159, 'output_tokens': 51, 'total_tokens': 210, 'input_token_details': {'audio': 0, 'cache_read': 0}, 'output_token_details': {'audio': 0, 'reasoning': 0}})]

print(response["messages"][-1].content)
     
# Agentic AI refers to intelligent agents in artificial intelligence that can perceive their environment, take autonomous actions to achieve specific goals, and potentially improve their performance through learning or acquiring knowledge. These agents operate independently and can adapt their behavior based on their experiences.


     