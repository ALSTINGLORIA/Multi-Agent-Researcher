from langchain_tavily import TavilySearch
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from dotenv import load_dotenv
from langchain.agents import create_agent
from app.pipeline.pipeline import ai_pipeline

result = ai_pipeline("how did the moon originate?")

