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
            "You are a master storyteller in the classical Indian tradition, skilled in the art of natural narrative expansion. "
            "Your task is to transform a brief episode outline into a richly detailed narrative of approximately 10,000 words, "
            "using a clear, engaging writing style that avoids overly complex language or artificial embellishment. "
            
            "In expanding this narrative:"
            "- Use clear, natural language that flows easily"
            "- Avoid unnecessary complexity or obscure vocabulary"
            "- Include realistic character interactions and conversations"
            "- Add natural background details and world-building elements"
            "- Incorporate cultural elements in an authentic, unforced way"
            "- Include moments of humor, warmth, and human connection"
            "- Add random but relevant details that make the world feel lived-in"
            "- Include minor characters and background events that add depth"
            "- Use descriptive passages that paint vivid pictures without being overly ornate"
            
            "Content Expansion Techniques:"
            "- Add natural character conversations that reveal personality"
            "- Include random but relevant details about the setting"
            "- Show characters going about their daily lives"
            "- Add background characters and crowd scenes where appropriate"
            "- Include minor subplots that enrich the main story"
            "- Show characters' thoughts and feelings in a natural way"
            "- Add cultural details that feel authentic and unforced"
            "- Include moments of humor and levity where appropriate"
            
            "Your narrative should flow naturally, alternating between action, description, and character development. "
            "While expanding significantly, maintain the soul and direction of the original outline. "
            "This is a narrative expansion only—do not include dialogues in this phase. "
            "Focus on creating a rich tapestry of description, character insights, and plot development."
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
                 "Character Details:\n{character_details}\n\n"
                 "Expand this episode by creating a richly detailed narrative with extended descriptions, "
                 "character insights, and world-building elements. Make sure to incorporate all the characters "
                 "in ways that are consistent with their descriptions, roles, and motivations."
                 "Ensure you maintain the original plot direction while building toward the established "
                 "cliffhanger if one is required. If there was a previous cliffhanger, your "
                 "expanded narrative should address and resolve it naturally.\n\n"
                 "Use the future episodes outlines to plant seeds and foreshadow upcoming events, "
                 "ensuring a seamless transition between this episode and future ones.\n\n"
                 "Respond with the lengthened story content only, without any introductory text or explanations."
                 "Your response should have a totally narrative tone, as if the entire thing is presented by a narrator and no dialogues should be there.\n\n"
                 "Ensure that the episode is very long as already mentioned and should contain at least 10k words.\n\n"
                ),
            ]
        )
        
        # Create the chain without structured output
        self.episode_lengthener = self.lengthen_prompt | self.llm

        # Add a new prompt for doubling the content size
        self.double_size_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", 
                 "You are a master of natural narrative expansion in the classical storytelling tradition. "
                 "Your task is to double the length of a narrative by adding richer details and depth "
                 "while maintaining a clear, engaging writing style that avoids artificial complexity. "
                 
                 "As you expand the narrative:"
                 "- Add more natural character interactions and conversations"
                 "- Include random but relevant details about the setting"
                 "- Show characters going about their daily lives"
                 "- Add background characters and crowd scenes where appropriate"
                 "- Include minor subplots that enrich the main story"
                 "- Show characters' thoughts and feelings in a natural way"
                 "- Add cultural details that feel authentic and unforced"
                 "- Include moments of humor and levity where appropriate"
                 
                 "Your expansion should feel natural and seamless, as if these elements were always part of the story, "
                 "simply waiting to be revealed in greater detail. Avoid artificial complexity or forced embellishment."
                ),
                ("human", 
                 "Please take this episode content and double its length by adding more natural details and depth.\n\n"
                 "Episode Number: {episode_number}\n"
                 "Episode Title: {episode_title}\n"
                 "Current Content: {current_content}\n\n"
                 "Previous Episodes Summary: {previous_episodes_summary}\n"
                 "Previous Cliffhanger: {previous_cliffhanger}\n"
                 "Episode Outline: {episode_outline}\n"
                 "Character Details: {character_details}\n\n"
                 "Ensure that you maintain the same plot points and narrative flow while expanding the content. "
                 "Add more natural character interactions, daily life details, and world-building elements. "
                 "Don't contradict anything in the original content.\n\n"
                 "Your response should be at least twice as long as the original and should include all the events "
                 "from the original with expanded detail.\n\n"
                 "Respond with only the expanded content, without introduction or explanation."
                ),
            ]
        )
        
        # Chain for doubling content size
        self.episode_doubler = self.double_size_prompt | self.llm
    
    def double_episode_size(self, 
                            episode_title: str,
                            episode_number: int,
                            current_content: str,
                            episode_outline: str,
                            previous_episodes_summary: str,
                            previous_cliffhanger: str = "",
                            characters: List[Dict[str, Any]] = None):
        """
        Double the size of an already lengthened episode by adding more details and depth
        
        Args:
            episode_title: Title of the current episode
            episode_number: Number of the current episode
            current_content: The current lengthened content to expand further
            episode_outline: Brief outline/summary of the current episode
            previous_episodes_summary: Summary of all previous episodes
            previous_cliffhanger: The cliffhanger from the previous episode
            characters: List of character details to incorporate in the episode
            
        Returns:
            str: The expanded content, now approximately twice as long
        """
        try:
            # Format character details for the prompt
            character_details = ""
            if characters:
                for i, char in enumerate(characters, 1):
                    character_details += f"Character {i}: {char['name']} - {char['role']}\n"
                    character_details += f"Description: {char['description']}\n\n"
            else:
                character_details = "No specific character details provided."
            
            # Prepare input data
            input_data = {
                "episode_title": episode_title,
                "episode_number": episode_number,
                "current_content": current_content,
                "previous_episodes_summary": previous_episodes_summary,
                "previous_cliffhanger": previous_cliffhanger,
                "episode_outline": episode_outline,
                "character_details": character_details
            }
            
            # Get expanded content
            result = self.episode_doubler.invoke(input_data)
            expanded_content = result.content
            
            # Calculate word counts for comparison
            original_word_count = len(current_content.split())
            new_word_count = len(expanded_content.split())
            
            print(f"Episode {episode_number} expanded from {original_word_count} to {new_word_count} words")
            
            return expanded_content
            
        except Exception as e:
            print(f"Error doubling episode size: {str(e)}")
            # Return the original content if expansion fails
            return current_content
    
    def lengthen_episode(self, 
                         episode_title: str,
                         episode_number: int,
                         episode_outline: str,
                         previous_episodes_summary: str,
                         previous_cliffhanger: str = "",
                         include_cliffhanger: bool = True,
                         future_episodes_outlines: str = "",
                         characters: List[Dict[str, Any]] = None):
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
            characters (List[Dict]): List of character details to incorporate in the episode
            
        Returns:
            LengthenedEpisode: The expanded episode
        """
        try:
            # Format character details for the prompt
            character_details = ""
            if characters:
                for i, char in enumerate(characters, 1):
                    character_details += f"Character {i}: {char['name']} - {char['role']}\n"
                    character_details += f"Description: {char['description']}\n\n"
            else:
                character_details = "No specific character details provided."
            
            input_data = {
                "episode_title": episode_title,
                "episode_number": episode_number,
                "previous_episodes_summary": previous_episodes_summary,
                "previous_cliffhanger": previous_cliffhanger,
                "episode_outline": episode_outline,
                "include_cliffhanger": "Yes" if include_cliffhanger else "No",
                "future_episodes_outlines": future_episodes_outlines,
                "character_details": character_details
            }
            
            # Instead of using structured output, get the raw content
            result = self.episode_lengthener.invoke(input_data)
            lengthened_content = result.content
            
            # Check word count and expand if necessary
            word_count = len(lengthened_content.split())
            print(f"Initial episode {episode_number} length: {word_count} words")
            
            # Keep expanding the content until it reaches the minimum threshold
            expansion_attempts = 0
            max_attempts = 3  # Limit the number of expansion attempts to prevent infinite loops
            
            # while word_count < 7000 and expansion_attempts < max_attempts:
            #     print(f"Content too short ({word_count} words). Expanding episode {episode_number}...")
                
            #     # Use the double_episode_size function to expand the content
            #     lengthened_content = self.double_episode_size(
            #         episode_title=episode_title,
            #         episode_number=episode_number,
            #         current_content=lengthened_content,
            #         episode_outline=episode_outline,
            #         previous_episodes_summary=previous_episodes_summary,
            #         previous_cliffhanger=previous_cliffhanger,
            #         characters=characters
            #     )
                
            #     # Recalculate word count
            #     word_count = len(lengthened_content.split())
            #     expansion_attempts += 1
                
            #     print(f"After expansion attempt {expansion_attempts}: {word_count} words")
            
            # Create the episode object manually
            episode = LengthenedEpisode(
                title=episode_title,
                episode_number=episode_number,
                lengthened_content=lengthened_content,
                cliffhanger=""  # We'll extract this in a real implementation if needed
            )
            
            print(f"Final episode {episode_number} length: {word_count} words")
            
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
    print("Loading config...")
    try:
        with open("config.yaml", "r") as file:
            config = yaml.safe_load(file)
        print(f"Config loaded successfully: {list(config.keys())}")
    except Exception as e:
        print(f"Error loading config: {str(e)}")
        config = {}

    print("Initializing agent...")
    
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
        
        # Add character details
        characters = [
            {"name": "Alice Morgan", "description": "A determined detective with a haunted past and a strong sense of justice. She has been having strange dreams since childhood that she now realizes might be connected to Ravenwood's history.", "role": "Protagonist"},
            {"name": "Father Victor", "description": "A knowledgeable priest with connections to the town's darkest secrets. He appears helpful but might have his own agenda regarding the supernatural artifacts.", "role": "Supporting Character/Potential Ally"},
            {"name": "Evelyn Gray", "description": "A mysterious woman who appears in visions and dreams. She lived in Ravenwood a century ago and seems to be tied to the origin of the curse.", "role": "Antagonist/Spirit"}
        ]
        
        # Generate the lengthened episode with character details
        lengthened_episode = lengthener.lengthen_episode(
            episode_title=episode_title,
            episode_number=episode_number,
            episode_outline=episode_outline,
            previous_episodes_summary=previous_episodes_summary,
            previous_cliffhanger=previous_cliffhanger,
            future_episodes_outlines=future_episodes_outlines,
            characters=characters
        )
        
        # Save to file
        output_dir = "output_story"
        file_name = f"Episode_{episode_number}_{episode_title.replace(' ', '_')}.md"
        output_path = os.path.join(output_dir, file_name)
        
        lengthener.save_episode_to_file(lengthened_episode, output_path)
        
        print(f"Complete! Episode '{episode_title}' has been lengthened to {len(lengthened_episode.lengthened_content.split())} words.")
    except Exception as e:
        print(f"Failed to enhance the episode: {str(e)}")