from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from llm_api import llm_api
from pydantic import BaseModel, Field
import concurrent.futures
import re

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
    
    def _split_into_chunks(self, text, max_chunk_size=3000):
        """
        Split text into chunks of approximately max_chunk_size words while preserving paragraphs.
        
        Args:
            text (str): Text to split
            max_chunk_size (int): Maximum number of words per chunk
            
        Returns:
            list: List of text chunks
        """
        # Split by paragraph breaks
        paragraphs = re.split(r'\n\n+', text)
        
        chunks = []
        current_chunk = []
        current_word_count = 0
        
        for paragraph in paragraphs:
            # Count words in paragraph
            paragraph_word_count = len(paragraph.split())
            
            # Check if adding this paragraph would exceed the chunk size
            if current_word_count + paragraph_word_count > max_chunk_size and current_chunk:
                # Save the current chunk and start a new one
                chunks.append('\n\n'.join(current_chunk))
                current_chunk = [paragraph]
                current_word_count = paragraph_word_count
            else:
                # Add paragraph to the current chunk
                current_chunk.append(paragraph)
                current_word_count += paragraph_word_count
        
        # Add the last chunk if it's not empty
        if current_chunk:
            chunks.append('\n\n'.join(current_chunk))
        
        return chunks
    
    def _translate_chunk(self, chunk, target_language):
        """
        Translate a single chunk of text.
        
        Args:
            chunk (str): Text chunk to translate
            target_language (str): Target language for translation
            
        Returns:
            str: Translated text chunk
        """
        try:
            result = self.translator.invoke({
                "target_language": target_language,
                "text": chunk
            })
            return result.translated_text
        except Exception as e:
            print(f"Error translating chunk: {str(e)}")
            return f"[Translation error: {str(e)}]"
        
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
        
        # Split the story into chunks
        chunks = self._split_into_chunks(story_text)
        print(f"Split story into {len(chunks)} chunks")
        
        translated_chunks = []
        
        # Translate chunks in parallel
        with concurrent.futures.ThreadPoolExecutor(max_workers=12) as executor:
            # Create a dictionary to track future objects and their corresponding chunk index
            future_to_index = {
                executor.submit(self._translate_chunk, chunk, target_language): i 
                for i, chunk in enumerate(chunks)
            }
            
            # Process translation results as they complete
            for future in concurrent.futures.as_completed(future_to_index):
                index = future_to_index[future]
                try:
                    translated_chunk = future.result()
                    print(f"Completed chunk {index+1}/{len(chunks)}")
                    # Store the result with its index for proper ordering
                    translated_chunks.append((index, translated_chunk))
                except Exception as e:
                    print(f"Error translating chunk {index}: {str(e)}")
                    translated_chunks.append((index, f"[Translation error in chunk {index+1}: {str(e)}]"))
        
        # Sort the chunks by original index and join them
        translated_chunks.sort(key=lambda x: x[0])
        translated_text = "\n\n".join(chunk for _, chunk in translated_chunks)
        
        print(f"Translation completed.")
        return translated_text


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
