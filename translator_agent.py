from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from llm_api import llm_api
from pydantic import BaseModel, Field

load_dotenv()

class TranslationResult(BaseModel):
    """Result of translating text to another language."""
    
    translated_text: str = Field(
        ...,
        description="The translated text in the target language.",
    )

class TranslatorAgent:
    """Agent for translating story content into different languages."""
    
    def __init__(self, api_key=None, model_type="translation"):
        """
        Initialize the translator agent.
        
        Args:
            api_key (str, optional): API key for the LLM service
            model_type (str, optional): Type of model to use for translation
        """
        # Initialize LLM using the same llm_api as in outline_generation_agent
        self.llm = llm_api(api_key=api_key, model_type=model_type)
        
        # Create structured output parser
        self.structured_llm_translation = self.llm.with_structured_output(TranslationResult)
        
        # Create translation prompt
        self.system_prompt = """You are an expert translator.
        Your task is to translate text to the requested target language.
        Maintain the original paragraph structure, formatting, and preserve any titles or headings.
        Ensure the translation sounds natural in the target language while preserving the original meaning.
        The expressions and idioms should be culturally appropriate for the target audience. Moreover the language must conform to the literature style of the target language.
        Return only the translated text in the 'translated_text' field."""
        
        self.translation_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", self.system_prompt),
                ("human", "Translate the following text to {target_language}:\n\n{text}"),
            ]
        )
        
        # Create translator chain
        self.translator = self.translation_prompt | self.structured_llm_translation
        
    def translate_story(self, story_text: str, target_language: str) -> str:
        """
        Translate a complete story to the target language.
        
        Args:
            story_text (str): Full story text to translate
            target_language (str): Target language (e.g., "Hindi", "Spanish", "French")
            
        Returns:
            str: Translated story
        """
        print(f"---TRANSLATING STORY TO {target_language.upper()}---")
        story_text = story_text[:5000]
        
        result = self.translator.invoke({
            "target_language": target_language,
            "text": story_text
        })
        
        print(f"Translation completed.")
        return result.translated_text


if __name__ == "__main__":
    # Example usage
    translator = TranslatorAgent()
    
    # Example story to translate
    story = """
    The Haunted Hotel
    
    A family arrives at an isolated hotel for a vacation. The place seemed perfect at first glance,
    with its grand architecture and beautiful surroundings. But as night fell, strange noises began 
    to disturb their sleep. The daughter started seeing figures that nobody else could see, 
    describing them as "the lost people." The father discovered the hotel's dark history of 
    disappearances. The family made a desperate attempt to escape as supernatural events escalated.
    """
    
    translated_story = translator.translate_story(story, "Bengali")
    print("\nOriginal Story:", story)
    print("\nTranslated Story:", translated_story)
