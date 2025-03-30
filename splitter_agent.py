import os
from dotenv import load_dotenv  
load_dotenv()
from typing import List, Dict, Any
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from llm_api import llm_api

class Episode(BaseModel):
    number: int = Field(..., description="The sequential number of the episode.")
    title: str = Field(..., description="A short title capturing the main idea of the episode.")
    content: str = Field(..., description="A detailed narrative of the episode, developing and advancing the plot.")
    cliffhanger: str = Field("", description="For every episode except the final one, provide a dramatic cliffhanger. For the final episode, leave this empty.")
    
    def to_dict(self):
        """Return a dictionary representation of the episode for JSON serialization"""
        return {
            "number": self.number,
            "title": self.title,
            "content": self.content,
            "cliffhanger": self.cliffhanger
        }

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
    def __init__(self, api_key=None):
        self.llm = llm_api(api_key=api_key, model_type="story_splitting")
        self.system_prompt = (
            "You are a master storyteller. Your task is to split a long narrative into interconnected episodes. "
            "Given an overall outline, specific plot points, and detailed character descriptions, generate a long, engaging story divided into episodes. "
            "For each episode, create a descriptive title, a detailed content section that moves the story forward, "
            "and a dramatic cliffhanger that leaves the reader eager to continue, except for the final episode. "
            "The final episode should have an empty cliffhanger field."
        )
        
        # Modified: Use a simpler approach to avoid function calling issues
        self.llm_regular = self.llm
        
        # Create a simpler prompt without relying on structured output
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
                 " - Episode Number: (the sequence number starting at 1)\n" 
                 " - Episode Title: (a catchy title that captures the essence)\n"
                 " - Episode Content: (a detailed narrative developing the plot)\n"
                 " - Cliffhanger: (a dramatic ending for all episodes except the final one)\n\n"
                 "Format your response as a series of episodes with clearly labeled NUMBER, TITLE, CONTENT, and CLIFFHANGER sections.\n\n"
                 "Ensure the episodes connect smoothly from one to the next. The final episode should NOT have a cliffhanger."
                ),
            ]
        )
        
        # Create the chain without structured output
        self.story_splitter = self.split_prompt | self.llm_regular

    def split_story(self, outline: str, plots: List[str], characters: List[dict], num_episodes: int = 5):
        input_data = {
            "outline": outline,
            "plots": plots,
            "characters": characters,
            "num_episodes": num_episodes,
        }
        
        # Get the raw response and parse it
        result = self.story_splitter.invoke(input_data)
        response_text = result.content if hasattr(result, 'content') else str(result)
        
        # Parse the response into episode objects
        episodes = self._parse_episodes(response_text, num_episodes)
        
        print("---GENERATED EPISODES ---")
        for episode in episodes:
            print(f"Episode {episode.number}: {episode.title}")
            print("Content:")
            print(episode.content)
            if episode.cliffhanger.strip():
                print("Cliffhanger:")
                print(episode.cliffhanger)
            print("------")
        
        return episodes
    
    def _parse_episodes(self, text: str, num_episodes: int) -> List[Episode]:
        """Parse the LLM response into Episode objects."""
        episodes = []
        
        # Split the text into episode chunks
        import re
        episode_chunks = re.split(r'Episode\s+\d+:|NUMBER:\s*\d+', text)[1:]  # Skip the first split which is empty
        
        # If we didn't get episode chunks, try alternative parsing
        if not episode_chunks and "EPISODE" in text.upper():
            episode_chunks = re.split(r'EPISODE\s+\d+:', text)[1:]
        
        # Process each chunk
        for i, chunk in enumerate(episode_chunks[:num_episodes], 1):
            # Extract the title
            title_match = re.search(r'(?:TITLE:|Episode Title:)\s*(.*?)(?:\n|$)', chunk, re.IGNORECASE)
            title = title_match.group(1).strip() if title_match else f"Episode {i}"
            
            # Extract the content
            content_match = re.search(r'(?:CONTENT:|Episode Content:)\s*(.*?)(?:(?:CLIFFHANGER:|Cliffhanger:)|$)', 
                                    chunk, re.IGNORECASE | re.DOTALL)
            content = content_match.group(1).strip() if content_match else chunk.strip()
            
            # Extract the cliffhanger if any
            cliffhanger_match = re.search(r'(?:CLIFFHANGER:|Cliffhanger:)\s*(.*?)(?:$)', chunk, re.IGNORECASE | re.DOTALL)
            cliffhanger = cliffhanger_match.group(1).strip() if cliffhanger_match else ""
            
            # Create and add the episode
            episode = Episode(
                number=i,
                title=title,
                content=content,
                cliffhanger="" if i == num_episodes else cliffhanger  # Ensure last episode has no cliffhanger
            )
            episodes.append(episode)
        
        # If we couldn't parse properly, create basic episodes
        if not episodes:
            print("Warning: Could not parse episodes properly. Creating default episodes.")
            for i in range(1, num_episodes + 1):
                episode = Episode(
                    number=i,
                    title=f"Episode {i}",
                    content=f"Content for episode {i}. " + text[:min(100, len(text))],
                    cliffhanger="" if i == num_episodes else "To be continued..."
                )
                episodes.append(episode)
        
        return episodes


if __name__ == "__main__":
    splitter = StorySplitterAgent()
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