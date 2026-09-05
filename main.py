from langchain_tavily import TavilySearch
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from dotenv import load_dotenv
from langchain.agents import create_agent

load_dotenv()

search_tool = TavilySearch(max_results = 1)

model = ChatNVIDIA(
    model="nvidia/nemotron-3.5-lightning-30b-a3b"
)

agent = create_agent(
    model = model,
    tools = [search_tool],
    system_prompt = (
        "You are an advanced AI assistant. "
        "Use the Tavily search tool to query real-time information when necessary."
    )
)

response = agent.invoke(
    {
        "messages" : [
            {
                "role" : "user",
                "content" : "Whats the current inr to usd converion rate"
            }
        ]
    }
)

print(response["messages"][-1].content)