from typing import List, Dict, Any
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from llm_api import llm_api, get_model_from_config
import json
import re

class CharacterDevelopmentAgent:
    def __init__(self, api_key=None):
        # Get the model from config
        self.model_type = "character_development"
        self.llm = llm_api(api_key=api_key, model_type=self.model_type)
        
        self.system_prompt = """You are an expert at developing story characters.
        
        Given a plot, generate detailed descriptions and roles for 3-5 distinctive Indian characters in the story.
        
        For each character, include:
        1. A unique and fitting name (name)
        2. A detailed physical and psychological description (description)
        3. The character's role in the story's plot (role)
        
        Format your response as a JSON array of character objects, each with 'name', 'description', and 'role' fields.
        """
        
        # Get the selected model name from config
        self.model_name = get_model_from_config(self.model_type)
        
        self.character_prompt = ChatPromptTemplate.from_messages([
            ("system", self.system_prompt),
            ("human", """Generate 3-5 character descriptions for the following plot.
            
Plot: {plot}

Format each character as a JSON object with 'name', 'description', and 'role' fields.

Example response format:
[
    {{"name": "Character Name", "description": "Character description", "role": "Character role"}},
    {{"name": "Another Character", "description": "Another description", "role": "Another role"}}
]""")
        ])
        
        self.character_generator = self.character_prompt | self.llm

    def generate_characters(self, plot):
        print(f"Generating characters for plot: {plot[:50]}...")
        
        try:
            # Use direct generation approach
            response = self.character_generator.invoke({"plot": plot})
            content = response.content if hasattr(response, 'content') else str(response)
            
            # Extract JSON from the response
            json_pattern = r'\[[\s\S]*?\]'
            json_match = re.search(json_pattern, content)
            
            if json_match:
                characters = json.loads(json_match.group())
                print("---GENERATED CHARACTER DESCRIPTIONS AND ROLES---")
                for character in characters:
                    print(f"Name: {character['name']}")
                    print(f"Description: {character['description']}")
                    print(f"Role: {character['role']}")
                    print("------")
                return characters
            else:
                # Try to extract individual JSON objects if we couldn't get an array
                char_pattern = r'\{[^{}]*"name":[^{}]*"description":[^{}]*"role":[^{}]*\}'
                matches = re.findall(char_pattern, content.replace("\n", " "))
                
                if matches:
                    characters = []
                    for match in matches:
                        try:
                            # Clean up the JSON string and make it valid
                            cleaned = match.replace("'", "\"")
                            char = json.loads(cleaned)
                            if 'name' in char and 'description' in char and 'role' in char:
                                characters.append(char)
                        except:
                            continue
                    
                    if characters:
                        print("---GENERATED CHARACTER DESCRIPTIONS AND ROLES---")
                        for character in characters:
                            print(f"Name: {character['name']}")
                            print(f"Description: {character['description']}")
                            print(f"Role: {character['role']}")
                            print("------")
                        return characters
                
                # If all parsing fails
                print("Could not parse characters from response, using fallback characters")
                return [
                    {"name": "Character 1", "description": "A character from the story", "role": "Protagonist"},
                    {"name": "Character 2", "description": "Another character from the story", "role": "Supporting"},
                    {"name": "Character 3", "description": "A third character from the story", "role": "Antagonist"}
                ]
                
        except Exception as e:
            print(f"Error generating characters: {str(e)}")
            return [
                {"name": "Character 1", "description": "A character from the story", "role": "Protagonist"},
                {"name": "Character 2", "description": "Another character from the story", "role": "Supporting"},
                {"name": "Character 3", "description": "A third character from the story", "role": "Antagonist"}
            ]


if __name__ == "__main__":
    agent = CharacterDevelopmentAgent()
    print("Enter the plot: ")
    plot = input('> ')

    characters = agent.generate_characters(plot)
    print("Generated Characters:", characters)