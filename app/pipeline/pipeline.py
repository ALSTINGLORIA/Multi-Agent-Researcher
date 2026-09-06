from app.agents.agent import search_agent, scrape_agent, writer_chain, critic_chain

state = {}

def ai_pipeline(topic : str) -> dict:
    search_agent_call = search_agent()
    search_result = search_agent_call.invoke(
        {
            "messages" : [("user",f"find a url of a website about {topic} and then only give the url back")]
        }
    )
    state['search_result'] = search_result['messages'][-1].content
    print(f"research --> {state['search_result']}")

    scrape_agent_call = scrape_agent()
    scrape_result = scrape_agent_call.invoke(
        {
            "messages" : [("user",f"based on the given url scrape the website and gain info. Url:{state['search_result']}")]
        }
    )
    state['scrape_result'] = scrape_result['messages'][-1].content
    print(f"scrape --> {state['scrape_result']}")

    state['writer'] = writer_chain.invoke(
        {
            "input" : topic,
            "research_data" : state['scrape_result']
        }
    )

    print(f"Writer --> {state['writer']}")

    state['critic'] = critic_chain.invoke(
        {
            "input" : topic,
            "content_to_review" : state['writer'],
            "research_data" : state['scrape_result']
        }
    )

    print(f"critic --> {state['critic']}")
    return state

