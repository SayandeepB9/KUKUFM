import os
from dotenv import load_dotenv  
load_dotenv()
from typing import List
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq

class Episode(BaseModel):
    number: int = Field(..., description="The sequential number of the episode.")
    title: str = Field(..., description="A short title capturing the main idea of the episode.")
    content: str = Field(..., description="A detailed narrative of the episode, developing and advancing the plot.")
    cliffhanger: str = Field("", description="For every episode except the final one, provide a dramatic cliffhanger. For the final episode, leave this empty.")

class SplitStoryQuery(BaseModel):
    outline: str = Field(..., description="A general overview of the story's theme and direction.")
    plots: List[str] = Field(..., description="A list of key plot points that should guide the narrative.")
    characters: List[dict] = Field(
        ..., 
        description="A list of character objects. Each object should contain at least 'name', 'description', and 'role'."
    )
    num_episodes: int = Field(5, description="The total number of episodes to divide the story into.")
    episodes: List[Episode] = Field(default_factory=list, description="The generated episodes that model the long story.")

class StorySplitterAgent:
    def __init__(self, model: str, api_key: str):
        self.llm = ChatGroq(model=model, api_key=api_key)
        self.system_prompt = (
            "You are a master storyteller. Your task is to split a long narrative into interconnected episodes. "
            "Given an overall outline, specific plot points, and detailed character descriptions, generate a long, engaging story divided into episodes. "
            "For each episode, create a descriptive title, a detailed content section that moves the story forward, "
            "and a dramatic cliffhanger that leaves the reader eager to continue, except for the final episode. "
            "The final episode should have an empty cliffhanger field."
        )
        self.structured_llm_split = self.llm.with_structured_output(SplitStoryQuery)
        self.split_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", self.system_prompt),
                ("human", 
                 "Using the provided inputs, generate a story split into {num_episodes} episodes.\n\n"
                 "Input Details:\n"
                 "Outline: {outline}\n"
                 "Plot Points: {plots}\n"
                 "Characters: {characters}\n\n"
                 "For each episode, please provide:\n"
                 " - 'number': The episode sequence number starting at 1.\n"
                 " - 'title': A catchy title that captures the episode's essence.\n"
                 " - 'content': A detailed narrative that develops the plot and integrates the given outline, plot points, and character details.\n"
                 " - 'cliffhanger': A dramatic cliffhanger ending for all episodes except the final one. The final episode should have an empty cliffhanger.\n\n"
                 "Ensure the episodes connect smoothly from one to the next."
                ),
            ]
        )
        self.story_splitter = self.split_prompt | self.structured_llm_split

    def split_story(self, outline: str, plots: List[str], characters: List[dict], num_episodes: int = 5):
        input_data = {
            "outline": outline,
            "plots": plots,
            "characters": characters,
            "num_episodes": num_episodes,
        }
        split_story = self.story_splitter.invoke(input_data)
        print("---GENERATED EPISODES ---")
        for episode in split_story.episodes:
            print(f"Episode {episode.number}: {episode.title}")
            print("Content:")
            print(episode.content)
            if episode.cliffhanger.strip():
                print("Cliffhanger:")
                print(episode.cliffhanger)
            print("------")
        return split_story.episodes


if __name__ == "__main__":
    model = "llama3-70b-8192"  
    api_key = os.getenv("GROQ_API_KEY")

    splitter = StorySplitterAgent(model=model, api_key=api_key)
    outline_text = "A mysterious struggle between light and dark unfolds in a haunted city where supernatural forces battle unseen foes."
    plots_list = [
        "The protagonists discover a hidden curse that links their past.",
        "An ancient secret society emerges from the shadows.",
        "Unexpected alliances and betrayals complicate the battle between good and evil.",
        "A groundbreaking revelation alters the course of the conflict.",
        "The final showdown culminates in a twist that redefines morality."
    ]
    characters_list = [
        {"name": "Alice", "description": "A determined detective haunted by her past, driven by justice.", "role": "Protagonist"},
        {"name": "Victor", "description": "A conflicted priest caught between faith and duty, harboring deep secrets.", "role": "Supporting"},
        {"name": "Evelyn", "description": "A spectral figure with an enigmatic past, whose guidance is as eerie as it is pivotal.", "role": "Antagonist/Guide"}
    ]
    episodes = splitter.split_story(outline_text, plots_list, characters_list, num_episodes=5)
    print("Generated Episodes:", episodes)