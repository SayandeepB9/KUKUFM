from typing import List, Dict, Any
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from llm_api import llm_api


class CharacterQuery(BaseModel):
    """Generate character descriptions and roles for a story given a plot."""

    plot: str = Field(
        ...,
        description="The plot of the story for which the characters should be developed.",
    )
    characters: List[Dict[str, str]] = Field(
        default_factory=list,                     
        description="A list of character dictionaries. Each character should have 'name', 'description', and 'role' keys.",
    )


class CharacterDevelopmentAgent:
    def __init__(self, model, api_key):
        self.llm = llm_api(model=model, api_key=api_key)
        self.system_prompt = """You are an expert at developing story characters.
        
        Given a plot, generate detailed descriptions and roles for 3-5 distinctive indian characters in the indian background in the story.
        
        For each character, include:
        1. A unique and fitting name (name)
        2. A detailed physical and psychological description (description)
        3. The character's role in the story's plot (role)
        
        Format your response so that each character is a dictionary with 'name', 'description', and 'role' keys
        in the 'characters' field of the response.
        """
        self.structured_llm_characters = self.llm.with_structured_output(CharacterQuery)
        self.character_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", self.system_prompt),
                ("human", "Develop characters for the following plot: {plot}. Return a list of character dictionaries with 'name', 'description', and 'role' keys in the 'characters' field."),
            ]
        )
        self.character_generator = self.character_prompt | self.structured_llm_characters

    def generate_characters(self, plot):
        character_details = self.character_generator.invoke({"plot": plot})
        print("---GENERATED CHARACTER DESCRIPTIONS AND ROLES---")
        for character in character_details.characters:
            print(f"Name: {character['name']}")
            print(f"Description: {character['description']}")
            print(f"Role: {character['role']}")
            print("------")
        return character_details.characters


if __name__ == "__main__":
    model = "llama-3.1-8b-instant" 
    api_key = "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"  

    agent = CharacterDevelopmentAgent(model=model, api_key=api_key)
    plot = "A group of friends spend a night in a haunted hotel and uncover its dark secrets."
    characters = agent.generate_characters(plot)
    print("Generated Characters:", characters)