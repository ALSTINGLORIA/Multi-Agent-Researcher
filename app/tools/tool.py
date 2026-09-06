import os
from tavily import TavilyClient
from dotenv import load_dotenv
import trafilatura
from langchain.tools import tool

load_dotenv()

@tool
def web_search(query : str) -> str:
    """ Used for searching the web for any url related to some information """
    tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
    result = tavily_client.search(query = query,max_results = 1)
    return result['results'][-1]['url']

@tool
def web_scrape(url : str) -> str:
    """ Used for finding information in the url by scrapping data """
    downloaded_html = trafilatura.fetch_url(url)
    if not downloaded_html:
        return "Error: Failed to download the webpage."

    article_text = trafilatura.extract(downloaded_html)
    return article_text