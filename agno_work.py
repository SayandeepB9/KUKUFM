from agno.agent import Agent
from agno.models.openai import OpenAIChat
from agno.tools.duckduckgo import DuckDuckGoTools
from agno.tools.googlesearch import GoogleSearchTools

from dotenv import load_dotenv
import os
load_dotenv()

writer_agent = Agent(
    name="Writer Agent",
    role="Generate the outline of a story based on a provided topic",
    model=OpenAIChat(id="gpt-4o-mini"),
    tools=[GoogleSearchTools()],
    instructions="Given a topic, generate a detailed outline of a story that highlights major events pointwise. Give a bit detailed outlines.",
    show_tool_calls=True,
    markdown=True,
)

# Define the agent team with updated instructions as well
agent_team = Agent(
    team=[writer_agent],
    model=OpenAIChat(id="gpt-4o-mini"),
    instructions=["Generate a detailed outline of a story for the provided topic, listing major events in a pointwise format."],
    show_tool_calls=True,
    markdown=True,
)

# Invoke the agent to generate a story outline for a given topic
agent_team.print_response("Generate the outline of a story about one haunted hotel.")