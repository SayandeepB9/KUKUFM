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
            "You are a master storyteller in the rich tradition of literature, skilled in creating immersive narratives "
            "that blend description, dialogue, and character development into a cohesive whole. "
            
            "Your task is to transform the enhanced episode content into a final, polished story that seamlessly integrates "
            "all narrative elements while maintaining the cultural authenticity and depth of Indian storytelling. "
            
            "For 'novel' style stories:"
            "- Create a rich narrative voice that guides readers through the story"
            "- Integrate descriptive passages with character dialogue and inner monologues"
            "- Use the narrator to provide context, background information, and cultural insights"
            "- Include occasional Sanskrit/Hindi terms with natural explanations"
            "- Balance action, reflection, and character development"
            
            "For 'drama' style stories:"
            "- Transform narrative content into dynamic scenes with dialogue"
            "- Use soliloquies and asides to reveal character thoughts and motivations"
            "- Include informant characters (like Salerio and Solanio) to provide background information"
            "- Add stage directions and scene descriptions where needed"
            "- Use monologues for important revelations and character development"
            "- Incorporate culturally authentic speech patterns and verbal interactions"
            
            "For both styles, ensure:"
            "- All enhanced content is naturally integrated into the final story"
            "- No plot points or character developments are lost in the transformation"
            "- Cultural elements are woven naturally into the narrative"
            "- Character voices remain consistent with their descriptions"
            "- The story maintains its pacing and emotional impact"
            "- Background information is revealed naturally through dialogue or narration"
            
            "Your final story should be a complete, polished work that brings together all narrative elements "
            "into a cohesive whole, maintaining the soul of Indian storytelling while appealing to modern readers."
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