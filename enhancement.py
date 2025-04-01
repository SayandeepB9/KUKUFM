import os
import yaml
import json
from typing import List, Optional, Dict, Any
from langchain_core.prompts import ChatPromptTemplate
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from llm_api import llm_api

class LengthenedEpisode(BaseModel):
    """Model representing a lengthened episode"""
    title: str = Field(..., description="The title of the episode.")
    episode_number: int = Field(..., description="The sequential number of the episode.")
    lengthened_content: str = Field(..., description="The expanded content of the episode, reaching approximately 10k words.")
    cliffhanger: Optional[str] = Field(None, description="The cliffhanger for this episode, if any.")

class EpisodeLengtheningAgent:
    """Agent that takes episode outlines and expands them to approximately 10k words"""
    
    def __init__(self, api_key=None):
        print("Initializing LLM for story_enhancement task")
        self.llm = llm_api(api_key=api_key, model_type="story_enhancement")
        # Create a parser for the output
        self.parser = PydanticOutputParser(pydantic_object=LengthenedEpisode)
        
        self.system_prompt = (
            
            "REMEBER YOU HAVE TO MAKE IT OF 10k WORDS"
            "You are a master storyteller specializing in rich, detailed narratives. "
            "Your task is to significantly expand an episode outline into a complete story "
            "of approximately 10k words. You must maintain perfect continuity with "
            "previous episodes (as summarized) and ensure that any cliffhanger from the "
            "previous episode is properly addressed and resolved in your expanded narrative. "
            "You must also seamlessly set up elements that will be important in future episodes "
            "to maintain narrative coherence across the entire story arc. "
            "If this episode should end with a cliffhanger, make sure your expanded content "
            "builds properly toward it. "
            "Add depth through detailed scene descriptions, character development, "
            "and rich world-building, all while maintaining the original plot points "
            "and tone. Keep the narrative engaging throughout the extended length."
            "Don't add any dialogues in the story and make it a narrative tone. "
        )
        
        # Using a regular prompt without structured output
        self.lengthen_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", self.system_prompt),
                ("human", 
                 "Please expand the following episode outline to approximately 10k words while maintaining narrative coherence and continuity.\n\n"
                 "Episode Number: {episode_number}\n"
                 "Episode Title: {episode_title}\n"
                 "Previous Episodes Summary: {previous_episodes_summary}\n"
                 "Previous Cliffhanger: {previous_cliffhanger}\n"
                 "Episode Outline: {episode_outline}\n"
                 "Should End With Cliffhanger: {include_cliffhanger}\n"
                 "Future Episodes Outlines: {future_episodes_outlines}\n\n"
                 "Expand this episode by creating a richly detailed narrative with extended descriptions, "
                 "character insights, and world-building elements. "
                 "Ensure you maintain the original plot direction while building toward the established "
                 "cliffhanger if one is required. If there was a previous cliffhanger, your "
                 "expanded narrative should address and resolve it naturally.\n\n"
                 "Use the future episodes outlines to plant seeds and foreshadow upcoming events, "
                 "ensuring a seamless transition between this episode and future ones.\n\n"
                 "Respond with the lengthened story content only, without any introductory text or explanations."
                 "Your response should have a totally narrative tone, as if the entire thing is presented by a narrator and no dialogues should be there.\n\n"
                 "Ensure that the episode is very long as already mentioned and should contain atleast 10k words.\n\n"
                ),
            ]
        )
        
        # Create the chain without structured output
        self.episode_lengthener = self.lengthen_prompt | self.llm
    
    def lengthen_episode(self, 
                         episode_title: str,
                         episode_number: int,
                         episode_outline: str,
                         previous_episodes_summary: str,
                         previous_cliffhanger: str = "",
                         include_cliffhanger: bool = True,
                         future_episodes_outlines: str = ""):
        """
        Lengthen an episode based on its outline, previous content, and future episode outlines
        
        Args:
            episode_title (str): Title of the current episode
            episode_number (int): Number of the current episode
            episode_outline (str): Brief outline/summary of the current episode to be expanded
            previous_episodes_summary (str): Summary of all previous episodes
            previous_cliffhanger (str): The cliffhanger from the previous episode that needs resolution
            include_cliffhanger (bool): Whether this episode should end with a cliffhanger
            future_episodes_outlines (str): Outlines of future episodes to help maintain continuity
            
        Returns:
            LengthenedEpisode: The expanded episode
        """
        try:
            input_data = {
                "episode_title": episode_title,
                "episode_number": episode_number,
                "previous_episodes_summary": previous_episodes_summary,
                "previous_cliffhanger": previous_cliffhanger,
                "episode_outline": episode_outline,
                "include_cliffhanger": "Yes" if include_cliffhanger else "No",
                "future_episodes_outlines": future_episodes_outlines
            }
            
            # Instead of using structured output, get the raw content
            result = self.episode_lengthener.invoke(input_data)
            lengthened_content = result.content
            
            # Create the episode object manually
            episode = LengthenedEpisode(
                title=episode_title,
                episode_number=episode_number,
                lengthened_content=lengthened_content,
                cliffhanger=""  # We'll extract this in a real implementation if needed
            )
            
            word_count = len(lengthened_content.split())
            print(f"Episode {episode_number} lengthened to approximately {word_count} words")
            
            return episode
            
        except Exception as e:
            print(f"Error lengthening episode: {str(e)}")
            raise

    def save_episode_to_file(self, episode: LengthenedEpisode, output_path: str):
        """
        Save the lengthened episode to a file
        
        Args:
            episode (LengthenedEpisode): The lengthened episode
            output_path (str): Path where to save the episode
        """
        try:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(f"# Episode {episode.episode_number}: {episode.title}\n\n")
                f.write(episode.lengthened_content)
                if episode.cliffhanger:
                    f.write(f"\n\n## Cliffhanger\n\n{episode.cliffhanger}")
            
            print(f"Saved episode to {output_path}")
        except Exception as e:
            print(f"Error saving episode to file: {str(e)}")
            raise


if __name__ == "__main__":
    print("Loading config from C:\\Users\\sagni\\OneDrive\\Desktop\\kukufm\\KUKUFM\\config.yaml")
    try:
        with open("config.yaml", "r") as file:
            config = yaml.safe_load(file)
        print(f"Config loaded successfully: {list(config.keys())}")
    except Exception as e:
        print(f"Error loading config: {str(e)}")
        config = {}

    print("OpenAI client initialized successfully\n")
    
    try:
        # Initialize the agent
        lengthener = EpisodeLengtheningAgent()
        
        # Hardcoded example data
        episode_title = "The Awakening"
        episode_number = 3
        episode_outline = "Alice discovers a hidden room in the abandoned church that contains ancient texts about the curse. Victor reveals he knows more than he initially let on, and Evelyn makes a surprising appearance that changes their understanding of the supernatural forces at play."
        
        previous_episodes_summary = """
        Episode 1 - The Shadow's Call: Detective Alice Morgan investigates a series of bizarre occurrences in Ravenwood City. She meets Father Victor who claims the events are connected to an ancient curse. Together they witness a supernatural manifestation that defies explanation.
        
        Episode 2 - Whispers in the Dark: Alice and Victor research the history of Ravenwood, discovering that similar events occurred exactly 100 years ago. They are followed by mysterious figures and receive cryptic warnings. Alice begins experiencing visions that appear to be memories that aren't her own.
        """
        
        previous_cliffhanger = "As Alice turned the final page of the historical record, the lights flickered and a spectral figure appeared in the corner of the room. 'You're looking in the wrong places,' the apparition whispered, its voice eerily familiar. 'I've been waiting for you, Detective Morgan.'"
        
        # Add future episodes outlines to provide context for what comes next
        future_episodes_outlines = """
        Episode 4 - The Secret Society: Alice and Victor discover that there is a secret society in Ravenwood that has been protecting an ancient artifact for centuries. They must infiltrate the society to learn more about the curse and how to stop it.
        
        Episode 5 - The Final Confrontation: Alice confronts the leader of the secret society and learns that Evelyn is connected to the original curse. A final showdown takes place at the town's oldest building where the curse began.
        """
        
        # Generate the lengthened episode with future episodes context
        lengthened_episode = lengthener.lengthen_episode(
            episode_title=episode_title,
            episode_number=episode_number,
            episode_outline=episode_outline,
            previous_episodes_summary=previous_episodes_summary,
            previous_cliffhanger=previous_cliffhanger,
            future_episodes_outlines=future_episodes_outlines
        )
        
        # Save to file
        output_dir = "output_story"
        file_name = f"Episode_{episode_number}_{episode_title.replace(' ', '_')}.md"
        output_path = os.path.join(output_dir, file_name)
        
        lengthener.save_episode_to_file(lengthened_episode, output_path)
        
        print(f"Complete! Episode '{episode_title}' has been lengthened to {len(lengthened_episode.lengthened_content.split())} words.")
    except Exception as e:
        print(f"Failed to enhance the episode: {str(e)}")