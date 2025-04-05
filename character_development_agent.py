from typing import List, Dict, Any
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from llm_api import llm_api, get_model_from_config
import json
import re
import os

class Character(BaseModel):
    """A character in a story"""
    
    name: str = Field(
        ...,
        description="A unique and fitting name for the character"
    )
    description: str = Field(
        ...,
        description="A detailed physical and psychological description of the character"
    )
    role: str = Field(
        ...,
        description="The character's role in the story's plot (e.g., protagonist, antagonist, supporting)"
    )

class CharacterList(BaseModel):
    """A list of characters for a story"""
    
    characters: List[Character] = Field(
        default_factory=list,
        description="List of characters for the story, typically 3-5 distinctive characters"
    )

class CharacterDevelopmentAgent:
    def __init__(self, api_key=None):
        # Get the model from config
        self.model_type = "character_development"
        self.llm = llm_api(api_key=api_key, model_type=self.model_type)
        
        # Load character types from story elements library
        self.character_types = self.load_character_types()
        
        self.system_prompt = f"""You are an expert at developing complex characters in the tradition of classical Indian storytelling.
        
        Given a plot, generate detailed descriptions and roles for a rich cast of characters (15-20 characters) that reflect Indian cultural values, traditions, and archetypes.
        
        Character Types to Include:
        1. Main Characters (5-6):
           - Primary protagonist(s) with clear goals and motivations
           - Key antagonists with complex motivations
           - Important supporting characters who drive the main plot
        
        2. Secondary Characters (5-6):
           - Characters who appear in multiple episodes but aren't central
           - Characters who provide comic relief or emotional support
           - Characters who help advance subplots
        
        3. Informant Characters (3-4):
           - Characters like Salerio and Solanio who keep readers informed
           - Gossips, messengers, or observers who provide background information
           - Characters who appear briefly to share important information
        
        4. Cameo Characters (2-4):
           - Characters who appear in only one or two episodes
           - Characters who serve specific plot functions
           - Background characters who add depth to the world
        
        For each character, provide:
        1. Name: A culturally appropriate and meaningful name with Indian origins
        
        2. Detailed Description:
           - Physical appearance (height, build, features, style of dress)
           - Voice and speech patterns (accent, tone, manner of speaking)
           - Personality traits and quirks
           - Background and history
           - Cultural and social status
           - Relationships with other characters
        
        3. DETAILED Role Description (this must be elaborated extensively):
           - Primary function in the story (protagonist, antagonist, supporting, etc.)
           - Character archetype and literary role (e.g., hero, mentor, trickster)
           - Specific episodes they appear in and their function in those episodes
           - How they drive plot development or subplots
           - Their relationships with other characters and how those evolve
           - Their character arc, transformation, or journey through the story
           - Their narrative significance and thematic relevance
           - Special skills or knowledge they bring to the story
           - Moral or philosophical stance they represent
           - Source of conflict or resolution they provide
           
        Consider including characters that embody classical Indian archetypes:
        - The virtuous hero/heroine
        - The wise guru or mentor
        - The devoted companion
        - The clever strategist
        - The noble adversary with redeeming qualities
        - The transformation-seeking disciple
        - The comic relief with hidden wisdom
        
        You may also incorporate relevant character types from this list as appropriate:
        {', '.join(self.character_types)}
        
        Each character should have clear values, motivations rooted in Indian philosophies, and relatable human traits.
        Make the characters feel authentic to culture while being universal in their appeal.
        Return a list of character objects with 'name', 'description', and 'role' fields in the 'characters' array.
        """
        
        # Get the selected model name from config
        self.model_name = get_model_from_config(self.model_type)
        
        # Create structured output parser
        self.structured_llm_characters = self.llm.with_structured_output(CharacterList)
        
        self.character_prompt = ChatPromptTemplate.from_messages([
            ("system", self.system_prompt),
            ("human", """Generate characters descriptions for the following plot.
            
Plot: {plot}

Consider which character types would best serve this story from the list provided.
Create characters with depth, nuance, and clear motivations.
Ensure the characters feel authentic for the Indian context and cultural setting.

Please provide characters that fit well with this story and have unique personalities and backgrounds.
Return them in the 'characters' field as structured data.""")
        ])
        
        # Chain the prompt with the structured output LLM
        self.character_generator = self.character_prompt | self.structured_llm_characters
        
        # Add refinement prompt
        self.refine_prompt = ChatPromptTemplate.from_messages([
            ("system", self.system_prompt),
            ("human", """Here is the previously generated character output for the plot:
Plot: {plot}

Previous Characters:
{previous_characters}

Human Feedback:
{feedback}

Please refine the character descriptions and roles based on the above feedback.
Consider which character types would best serve this story from the list provided.
Create characters with depth, nuance, and clear motivations.

Return them in the 'characters' field as structured data.""")
        ])
        
        # Chain the refine prompt with the structured output LLM
        self.refine_generator = self.refine_prompt | self.structured_llm_characters
        
    def load_character_types(self):
        """Load character types from the story elements library JSON file"""
        try:
            # Find the story_elements.json file
            file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "story_elements.json")
            
            
            if not os.path.exists(file_path):
                file_path = "story_elements.json"
                
            with open(file_path, 'r') as f:
                story_elements = json.load(f)
                
            
            character_types = story_elements.get("literary_elements", {}).get("characters", [])
            
            
            if character_types:
                print(f"Loaded {len(character_types)} character types from story elements library")
                return character_types
            else:
                print("No character types found in story elements library")
                return self.get_default_character_types()
                
        except Exception as e:
            print(f"Error loading story elements: {str(e)}")
            return self.get_default_character_types()
            
    def get_default_character_types(self):
        """Return default character types if the JSON file cannot be loaded"""
        return [
            "Protagonist (hero/heroine)", 
            "Antagonist (villain)", 
            "Deuteragonist (secondary main character)",
            "Foil (character highlighting traits of another)",
            "Dynamic (evolving)",
            "Round (complex, multidimensional)",
            "Anti-hero",
            "Tragic hero",
            "Mentor/guide",
            "Sidekick/confidant",
            "Trickster"
        ]

    def generate_characters(self, plot):
        print(f"Generating characters for plot: {plot[:50]}...")
        
        try:
           
            response = self.character_generator.invoke({"plot": plot})
            
            characters = [
                {"name": char.name, "description": char.description, "role": char.role}
                for char in response.characters
            ]
            
            print("---GENERATED CHARACTER DESCRIPTIONS AND ROLES---")
            for character in characters:
                print(f"Name: {character['name']}")
                print(f"Description: {character['description']}")
                print(f"Role: {character['role']}")
                print("------")
            
            return characters
                
        except Exception as e:
            print(f"Error generating characters: {str(e)}")

            return [
                {"name": "Character 1", "description": "A character from the story", "role": "Protagonist"},
                {"name": "Character 2", "description": "Another character from the story", "role": "Supporting"},
                {"name": "Character 3", "description": "A third character from the story", "role": "Antagonist"}
            ]
    
    def refine_characters(self, plot: str, previous_characters: List[Dict[str, Any]], feedback: str) -> List[Dict[str, Any]]:
        """
        Refines character descriptions based on human feedback.

        Args:
            plot (str): The original plot of the story
            previous_characters (List[Dict[str, Any]]): The characters from the previous generation
            feedback (str): Human feedback on how to improve the characters

        Returns:
            List[Dict[str, Any]]: Refined list of character details
        """
        print(f"Refining characters based on feedback...")
        
        previous_characters_str = json.dumps(previous_characters, indent=2)
        
        try:
            
            response = self.refine_generator.invoke({
                "plot": plot,
                "previous_characters": previous_characters_str,
                "feedback": feedback
            })
            
            refined_characters = [
                {"name": char.name, "description": char.description, "role": char.role}
                for char in response.characters
            ]
            
            print("---REFINED CHARACTER DESCRIPTIONS AND ROLES---")
            for character in refined_characters:
                print(f"Name: {character['name']}")
                print(f"Description: {character['description']}")
                print(f"Role: {character['role']}")
                print("------")
            
            return refined_characters
                
        except Exception as e:
            print(f"Error refining characters: {str(e)}")
            print("Returning original characters without changes.")
            return previous_characters


if __name__ == "__main__":
    agent = CharacterDevelopmentAgent()
    print("Enter the plot: ")
    plot = input('> ')

    characters = agent.generate_characters(plot)
    print("Generated Characters:", characters)
    
    feedback = input("\nEnter your feedback to refine the generated characters (press enter to skip): ")
    if feedback.strip():
        refined_characters = agent.refine_characters(plot, characters, feedback)
        print("Refined Characters:", refined_characters)
    else:
        print("No feedback provided. Generated characters remain unchanged.")