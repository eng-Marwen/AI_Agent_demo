from dotenv import load_dotenv
import os

from langchain_groq import ChatGroq
from langchain.agents import create_agent
from pydantic import BaseModel

load_dotenv()


class ChatResponse(BaseModel):
    topic: str
    summary: str
    sources: list[str]
    tools_used: list[str]


def ai_agent():

    llm = ChatGroq(
        api_key=os.getenv("GROQ_API_KEY"),
        model="llama-3.3-70b-versatile",
        temperature=0.7,
        max_tokens=1000,
    )

    system_prompt = """
    You are a helpful research assistant.

    Answer user questions about research topics.
    Return a structured answer containing:
    - topic
    - summary
    - sources
    - tools_used
    """

    agent = create_agent(
        model=llm,
        tools=[],
        system_prompt=system_prompt
    )

    response = agent.invoke(
        {
            "messages": [
                (
                    "user",
                    "What is the impact of climate change on global food security?"
                )
            ]
        }
    )

    print(response)


if __name__ == "__main__":
    ai_agent()