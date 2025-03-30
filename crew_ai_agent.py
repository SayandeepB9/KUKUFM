from crewai import Agent, Task, Crew
from crewai_tools import SerperDevTool

# Your existing agent definition
research_agent = Agent(
    role="Research Analyst",
    goal="Find and summarize information about specific topics",
    backstory="You are an experienced researcher with attention to detail",
    verbose=True  # Enable logging for debugging
)

# Create a task for the agent
research_task = Task(
    description="Research the latest developments in artificial intelligence and summarize key findings",
    expected_output="A comprehensive summary of recent AI developments",
    agent=research_agent
)

# Create a crew with the agent and task
crew = Crew(
    agents=[research_agent],
    tasks=[research_task]
)

# Execute the crew and get the result
result = crew.kickoff()
print(result)