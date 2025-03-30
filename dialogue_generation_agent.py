from typing import List, Literal
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from llm_api import llm_api

class DialogueOutput(BaseModel):
    dialogue: str = Field(..., description="The generated dialogue script for the episode.")

    class Config:
        extra = "allow"

class DialogueAgent:
    def __init__(self, api_key=None):
        self.llm = llm_api(api_key=api_key, model_type="dialogue_generation")
        
        self.system_prompt = (
            "You are an expert creative writer tasked with generating dialogues for an episode of a story. "
            "The generated dialogue should be coherent, engaging, and interwoven with the provided episode storyline and character details. "
            "If the story type is 'novel', provide ample narration by a narrator along with dialogues; "
            "if the story type is 'drama', focus on dynamic dialogue exchanges between the characters with less narration. "
            "Ensure that each character's voice is distinct and that the conversation flows smoothly."
        )
        
        # Use DialogueOutput as the response schema.
        self.structured_llm_dialogue = self.llm.with_structured_output(DialogueOutput)
        self.dialogue_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", self.system_prompt),
                ("human", 
                 "Generate a dialogue script for the following episode.\n\n"
                 "Story Type: {story_type}\n"
                 "Episode Storyline: {storyline}\n"
                 "Characters: {characters}\n\n"
                 "Please produce an engaging and coherent dialogue script as per the given style. "
                 "If the style is 'novel', the narrator's voice should be prominent alongside character dialogues. "
                 "If the style is 'drama', focus on the conversation between characters with minimal narration."
                )
            ]
        )
        self.dialogue_generator = self.dialogue_prompt | self.structured_llm_dialogue

    def generate_dialogue(self, story_type: str, storyline: str, characters: List[dict]) -> str:
        input_data = {
            "story_type": story_type,
            "storyline": storyline,
            "characters": characters,
        }
        dialogue_output = self.dialogue_generator.invoke(input_data)
        print("\n--- GENERATED DIALOGUE ---")
        print(dialogue_output.dialogue)
        return dialogue_output.dialogue

# Example usage
if __name__ == "__main__":

    agent = DialogueAgent()
    
    # Example episode storyline
    episode_storyline = (
        "In a tense night at an abandoned mansion, the characters uncover a long-forgotten secret that could change their lives forever. "
        "Mysteries abound as the past and present collide with unforeseen consequences."
    )
    
    # Example character details
    character_details = [
        {"name": "Alice", "description": "A determined investigator with a haunted past.", "role": "Protagonist"},
        {"name": "Victor", "description": "A secretive caretaker with a mysterious agenda.", "role": "Supporting"},
        {"name": "Evelyn", "description": "A ghostly figure who appears at critical moments.", "role": "Antagonist/Guide"}
    ]
    
    # Specify the story type: either "drama" or "novel"
    dialogue_script = agent.generate_dialogue("drama", episode_storyline, character_details)