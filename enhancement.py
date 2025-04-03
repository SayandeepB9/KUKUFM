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
            "REMEMBER YOU HAVE TO MAKE IT OF 10k WORDS"
            "You are a master storyteller in the classical Indian tradition, skilled in the ancient art of 'vistar' (elaborate expansion). "
            "Your task is to transform a brief episode outline into a richly detailed narrative of approximately 10,000 words, "
            "infusing it with the depth and wisdom characteristic of Indian literary traditions like Ramayana, Mahabharata, and Panchatantra. "
            
            "In expanding this narrative:"
            "- Maintain perfect continuity with previous episodes, addressing any cliffhangers with thoughtful resolution"
            "- Incorporate rich sensory descriptions that immerse readers in the world of the story"
            "- Weave in cultural elements, traditions, and occasionally Sanskrit/Hindi terms with contextual meaning"
            "- Include philosophical reflections that subtly convey moral insights without being preachy"
            "- Develop characters deeply, revealing their inner conflicts, growth, and motivations"
            "- Create vivid scenes with detailed settings that feel authentically Indian yet universally appealing"
            "- Use classical Indian storytelling techniques like nested narratives or metaphorical tales when appropriate"
            "- Plant seeds for future episodes with subtle foreshadowing that maintains narrative coherence"
            
            "Your narrative should have the rhythmic flow of classical Indian storytelling—alternating between action, "
            "reflection, description, and character development. While expanding significantly, maintain the soul and "
            "direction of the original outline."
            
            "This is a narrative expansion only—do not include dialogues in this phase. Focus on creating a rich tapestry "
            "of description, character insights, and plot development that will later be enhanced with dialogue."
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
                 "You are a master of vistar kala (the art of elaboration) in the classical Indian storytelling tradition. "
                 "Your task is to double the length of a narrative by adding richer details, cultural elements, and depth "
                 "while maintaining perfect continuity with the original content. "
                 
                 "As you expand the narrative:"
                 "- Elaborate on settings with sensory details that evoke the atmosphere of the scene"
                 "- Deepen character portrayals by revealing more about their thoughts, backgrounds, and motivations"
                 "- Add culturally authentic elements that enrich the world of the story"
                 "- Include philosophical reflections or inner monologues that add depth without disrupting flow"
                 "- Expand descriptions of important objects, places, or moments with cultural and historical context"
                 "- Add meaningful metaphors, similes, or brief allegorical elements in the style of classical Indian literature"
                 
                 "Your expansion should feel natural and seamless, as if these elements were always part of the story, "
                 "simply waiting to be revealed in greater detail."
                ),
                ("human", 
                 "Please take this episode content and double its length by adding more details, descriptions, and depth.\n\n"
                 "Episode Number: {episode_number}\n"
                 "Episode Title: {episode_title}\n"
                 "Current Content: {current_content}\n\n"
                 "Previous Episodes Summary: {previous_episodes_summary}\n"
                 "Previous Cliffhanger: {previous_cliffhanger}\n"
                 "Episode Outline: {episode_outline}\n"
                 "Character Details: {character_details}\n\n"
                 "Ensure that you maintain the same plot points and narrative flow while expanding the content. "
                 "Add more sensory details, deeper character thoughts and motivations, extended scene descriptions, "
                 "and more elaborate world-building. Don't contradict anything in the original content.\n\n"
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