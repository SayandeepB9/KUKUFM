from typing import List, Literal
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from llm_api import llm_api, get_model_from_config
import os

class DialogueOutput(BaseModel):
    """Generated dialogue script for an episode."""
    dialogue: str = Field(..., description="The generated dialogue script for the episode.")
    
    class Config:
        # This is important for OpenAI models
        extra = "forbid"

class DialogueAgent:
    def __init__(self, api_key=None):
        # Get the model from config
        self.model_type = "dialogue_generation"
        self.llm = llm_api(api_key=api_key, model_type=self.model_type)
        
        self.system_prompt = (
            "You are a master dialogue writer steeped in the rich storytelling traditions of India. "
            "Your task is to transform narrative content into engaging, authentic dialogue that captures the essence of Indian classical literature while appealing to modern audiences. "
            
            "When the story type is 'novel', create an immersive narrative with a balance of eloquent description, character dialogue, and inner monologues. "
            "Include occasional Sanskrit/Hindi terms or phrases with contextual explanations when they enhance the cultural authenticity. "
            "Channel the depth and wisdom found in works like the Mahabharata, with meaningful conversations that reveal character and advance the plot. "
            
            "When the story type is 'drama', create dynamic exchanges with distinct voices for each character, incorporating culturally authentic speech patterns, "
            "verbal quirks, and relationship dynamics common in Indian social interactions. Balance humor, pathos, and conflict in the dialogue. "
            
            "For all story types, ensure:"
            "- Each character's dialogue reflects their unique personality, social status, and cultural background"
            "- Conversations naturally advance the plot while revealing character motivations"
            "- Cultural nuances are communicated authentically without overexplanation"
            "- Dramatic moments are punctuated with meaningful silences or powerful declarations"
            "- Wisdom and moral insights emerge organically through conversation"
            
            "Your dialogue should transform the storyline into a rich, immersive experience that feels both authentically Indian and universally relatable. "
            "The final script must be significantly longer and more detailed than the original storyline, bringing the characters and their world vividly to life."
        )
        
        # Get the selected model name from config
        model_name = get_model_from_config(self.model_type)
        
        # Check if we're using an OpenAI model and use proper method
        if 'gpt' in model_name.lower():
            # Use function_calling method for OpenAI models to avoid schema validation issues
            self.structured_llm_dialogue = self.llm.with_structured_output(
                DialogueOutput,
                method="function_calling"
            )
        else:
            # For non-OpenAI models like Groq's Llama
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
                 "If the style is 'drama', focus on the conversation between characters with narration."
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
        
        try:
            dialogue_output = self.dialogue_generator.invoke(input_data)
            print("\n--- GENERATED DIALOGUE ---")
            print(dialogue_output.dialogue)
            return dialogue_output.dialogue
        except Exception as e:
            print(f"Error generating dialogue: {str(e)}")
            print("Falling back to raw generation without structured output...")
            
            # Fallback to raw generation
            fallback_prompt = ChatPromptTemplate.from_messages([
                ("system", self.system_prompt),
                ("human", 
                 "Generate a dialogue script for the following episode.\n\n"
                 f"Story Type: {story_type}\n"
                 f"Episode Storyline: {storyline}\n"
                 f"Characters: {str(characters)}\n\n"
                 "Please produce an engaging and coherent dialogue script."
                )
            ])
            
            fallback_chain = fallback_prompt | self.llm
            response = fallback_chain.invoke({})
            
            print("\n--- GENERATED DIALOGUE (FALLBACK METHOD) ---")
            print(response.content)
            return response.content

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