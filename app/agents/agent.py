from app.tools.tool import web_search, web_scrape
from langchain.agents import create_agent
import os
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

model = ChatNVIDIA(
    model="nvidia/nemotron-3.5-lightning-30b-a3b",
    api_key = os.getenv("NVIDIA_API_KEY")
)

def search_agent():
    return create_agent(
        model = model,
        tools = [web_search]
    )

def scrape_agent():
    return create_agent(
        model = model,
        tools = [web_scrape]
    )

writer_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are an elite research agent and data synthesizer. Your primary goal is to analyze "
        "raw research data and transform it into a well-structured, objective, and comprehensive summary. "
        "Strictly adhere to the facts provided in the research data, avoid speculation, organize your response "
        "with clear headings or bullet points where appropriate, and highlight any key insights or consensus."
    ),
    (
        "user",
        "Topic to research: {input}\n\n"
        "Here is the raw research data gathered from the search tool:\n"
        "{research_data}"
    )
])

writer_chain = writer_prompt | model | StrOutputParser()

critic_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are an elite content critic, fact-checker, and editorial editor. Your primary goal is to rigorously "
        "evaluate drafts and research summaries for accuracy, clarity, depth, and bias. "
        "Identify any logical gaps, unsupported claims, or weak arguments. "
        "Provide constructive, actionable feedback and suggest specific improvements to elevate the quality of the content."
    ),
    (
        "user",
        "Please review and critique the following content regarding: {input}\n\n"
        "--- CONTENT TO REVIEW ---\n"
        "{content_to_review}\n\n"
        "--- ORIGINAL RESEARCH DATA (for fact-checking) ---\n"
        "{research_data}"
    )
])

critic_chain = critic_prompt | model | StrOutputParser()