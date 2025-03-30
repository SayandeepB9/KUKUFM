from langchain.prompts import ChatPromptTemplate
import os
import json
import datetime
import random
from typing import List, Dict, Any
from llm_api import llm_api

class StoryElementLibrary:
    """Library of story elements that provides plot options based on existing outlines."""
    
    def __init__(self, model_name="llama3-70b-8192", api_key=None):
        """Initialize the story element library."""
        # Set up API key
        if api_key:
            os.environ["GROQ_API_KEY"] = api_key
        elif "GROQ_API_KEY" not in os.environ:
            raise ValueError("GROQ API key must be provided either as an argument or as an environment variable")
        
        # Initialize LLM using the llm_api function
        self.llm = llm_api(
            model=model_name,
            api_key=api_key,
            streaming=False
        )
        
        # Basic story element categories
        self.plot_twists = {
            "general": [
                "The protagonist discovers they've been misled the entire time",
                "A trusted ally is revealed to be working against the protagonist",
                "The antagonist is revealed to be a family member",
                "A seemingly unimportant character is revealed as the mastermind",
                "Two timelines are revealed to be occurring simultaneously",
                "The entire story is revealed to be a dream/hallucination",
                "The protagonist and antagonist must work together to overcome a greater threat",
                "A character believed dead returns at a crucial moment",
                "A prophecy/prediction is fulfilled but in an unexpected way",
                "A character's true identity is revealed, changing the story's context"
            ],
            "ghost": [
                "The ghost is actually protecting the protagonist from a greater evil",
                "The protagonist discovers they've been dead the whole time",
                "The haunting is revealed to be a hoax by a living person with ulterior motives",
                "The ghost is actually from the future, not the past",
                "Multiple spirits are revealed to be different aspects of the same person",
                "The supposed ghost is actually a living person trapped between dimensions",
                "The protagonist discovers they can see ghosts due to a near-death experience in their forgotten past",
                "The ghost is revealed to be a manifestation of the protagonist's guilt or trauma",
                "The haunting is revealed to be caused by an object, not a location",
                "The ghost is revealed to be the protagonist from another timeline"
            ],
            "sci-fi": [
                "Technology intended to help humanity is revealed to have a sinister purpose",
                "The alien species is revealed to be evolved humans from the future",
                "The protagonist discovers they are a clone/android/synthetic human",
                "The apparently distant planet is revealed to be future Earth",
                "The antagonist is revealed to be the protagonist from an alternate timeline",
                "The technology is revealed to be powered by human consciousness/souls",
                "The mission is revealed to be a simulation/experiment",
                "The seemingly benevolent AI is revealed to have its own agenda",
                "The disease/phenomenon is revealed to be caused by time travel",
                "The corporation is revealed to be controlled by a non-human intelligence"
            ]
        }
        
        # Initialize the plot options prompt
        self.plot_options_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a creative writing assistant that specializes in providing plot options "
                     "based on a story outline. Generate 10 distinct and creative plot options that would work well "
                     "with the provided outline. Each plot option should be a complete plot point that could be "
                     "inserted into the story. Format your response as a JSON array of strings, where each string "
                     "is a single plot option. Make each option distinct and creative."),
            ("human", "Story Type: {story_type}\nOutline: {outline}\n\n"
                    "Please provide 10 engaging plot options that could enhance this story.")
        ])
        
        self.plot_options_chain = self.plot_options_prompt | self.llm
        
        # Store selected plot options
        self.selected_plot_options = []

    def generate_plot_options(self, outline, story_type):
        """Generate plot options that would enhance the outline."""
        try:
            print("\nGenerating plot options... (this may take a moment)\n")
            
            response = self.plot_options_chain.invoke({
                "story_type": story_type,
                "outline": outline
            })
            
            result_text = response.content if hasattr(response, 'content') else str(response)
            
            # Try to parse the response as JSON
            try:
                # Find JSON in the response (it might be surrounded by other text)
                import re
                json_match = re.search(r'(\[[\s\S]*\])', result_text)
                if json_match:
                    cleaned_json = json_match.group(1)
                    return json.loads(cleaned_json)
                else:
                    # If JSON pattern not found, try to parse the whole response
                    return json.loads(result_text)
            except json.JSONDecodeError:
                # If JSON parsing fails, try to extract plot options with regex
                print("Could not parse LLM response as JSON. Using fallback parsing method.")
                
                # Fallback: Try to extract numbered or bulleted items
                plot_options = []
                
                # Look for numbered items
                lines = result_text.split('\n')
                for line in lines:
                    # Remove numbers, bullets, etc. at the beginning of lines
                    import re
                    cleaned_line = re.sub(r'^[\d\-\*\.\s]+', '', line).strip()
                    if cleaned_line and len(cleaned_line) > 10:  # Minimum length to be a plot option
                        plot_options.append(cleaned_line)
                
                # Look for quoted strings
                if not plot_options:
                    quotes = re.findall(r'\"(.*?)\"', result_text)
                    for quote in quotes:
                        if len(quote) > 10:  # Minimum length to be a plot option
                            plot_options.append(quote)
                
                if plot_options:
                    return plot_options
                else:
                    print("Fallback parsing also failed. Using story element library.")
                    return self._create_fallback_plot_options(story_type)
                
        except Exception as e:
            print(f"Error generating plot options: {str(e)}")
            return self._create_fallback_plot_options(story_type)
    
    def _create_fallback_plot_options(self, story_type):
        """Create fallback plot options when LLM response parsing fails."""
        # Use the story element library to generate plot options
        if story_type in self.plot_twists:
            return self.plot_twists[story_type]
        else:
            return self.plot_twists["general"]

    def display_plot_options(self, plot_options):
        """Display plot options in a numbered format."""
        print("\n=== AVAILABLE PLOT OPTIONS ===\n")
        
        for i, option in enumerate(plot_options, 1):
            print(f"Option {i}: {option}")
    
    def select_plot_options(self, plot_options):
        """Allow user to select plot options."""
        print("\n=== SELECT PLOT OPTIONS ===")
        print("Enter the numbers of the options you want to use, separated by commas.")
        print("For example, enter '1,3,5' to select options 1, 3, and 5.")
        
        while True:
            try:
                selection = input("Select options: ")
                
                # Handle empty input
                if not selection.strip():
                    print("No options selected. Please try again.")
                    continue
                
                # Parse the selection
                selected_indices = [int(idx.strip()) for idx in selection.split(',')]
                
                # Validate the selection
                valid_indices = []
                for idx in selected_indices:
                    if 1 <= idx <= len(plot_options):
                        valid_indices.append(idx)
                    else:
                        print(f"Invalid option {idx}. Ignoring.")
                
                if not valid_indices:
                    print("No valid options selected. Please try again.")
                    continue
                
                # Get the selected options
                selected_options = [plot_options[idx-1] for idx in valid_indices]
                self.selected_plot_options = selected_options
                
                return selected_options
                
            except ValueError:
                print("Please enter valid numbers separated by commas.")
    
    def save_selected_options(self, selected_options, filename=None):
        """Save the selected plot options to a file."""
        if not selected_options:
            print("No plot options to save.")
            return None
        
        if not filename:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"selected_plot_options_{timestamp}.json"
        
        try:
            with open(filename, "w", encoding="utf-8") as f:
                json.dump(selected_options, f, indent=2, ensure_ascii=False)
            print(f"\nSelected plot options saved to {filename}")
            return filename
        except Exception as e:
            print(f"Error saving plot options: {str(e)}")
            return None


def extract_outline_from_input(input_text):
    """Extract outline list from input text if necessary."""
    # Check if input is already a list
    if isinstance(input_text, list):
        return input_text
    
    # Try to parse as JSON
    try:
        parsed = json.loads(input_text)
        if isinstance(parsed, list):
            return parsed
    except:
        pass
    
    # Try to parse as string representation of list
    try:
        import ast
        parsed = ast.literal_eval(input_text)
        if isinstance(parsed, list):
            return parsed
    except:
        pass
    
    # Try to extract bracketed content
    import re
    match = re.search(r'\[(.*?)\]', input_text, re.DOTALL)
    if match:
        # Parse the bracketed content
        content = match.group(1)
        items = []
        for item in re.findall(r'\'(.*?)\'', content):
            items.append(item)
        if items:
            return items
    
    # Split by newlines and clean up
    lines = input_text.strip().split('\n')
    cleaned_lines = []
    for line in lines:
        # Remove leading numbers or bullets
        cleaned = re.sub(r'^[\d\.\-\*]+\s*', '', line).strip()
        if cleaned:
            cleaned_lines.append(cleaned)
    
    return cleaned_lines


def main():
    # Display header
    current_datetime = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    current_user = "Sag12345-IITKGP"
    
    print(f"Current Date and Time (UTC - YYYY-MM-DD HH:MM:SS formatted): {current_datetime}")
    print(f"Current User's Login: {current_user}")
    print("\n=== STORY PLOT OPTIONS SELECTOR ===")
    
    # Get API key if not set in environment
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        print("Please enter your Groq API key:")
        api_key = input("> ")
    
    try:
        # Initialize the story element library
        library = StoryElementLibrary(api_key=api_key)
        
        # Get the story type
        print("\nPlease specify the story type (e.g., ghost, sci-fi, fantasy, mystery, etc.):")
        story_type = input("> ").lower()
        
        # Get the outline
        print("\nPlease paste your story outline (in list format):")
        outline_input = input("> ")
        
        # Parse the outline from input
        outline = extract_outline_from_input(outline_input)
        
        if not outline:
            print("Could not parse a valid outline. Please check your input format.")
            return
        
        print("\nParsed Outline:")
        for i, point in enumerate(outline, 1):
            print(f"{i}. {point}")
        
        # Generate plot options
        plot_options = library.generate_plot_options(outline, story_type)
        
        if not plot_options:
            print("Could not generate plot options. Please try again.")
            return
        
        # Display plot options
        library.display_plot_options(plot_options)
        
        # Allow user to select options
        selected_options = library.select_plot_options(plot_options)
        
        # Display final selected options
        print("\n=== SELECTED PLOT OPTIONS ===")
        for i, option in enumerate(selected_options, 1):
            print(f"{i}. {option}")
        
        # Save the selected options
        library.save_selected_options(selected_options)
        
    except Exception as e:
        print(f"\nAn unexpected error occurred: {str(e)}")
        print("Please check your API key and internet connection, then try again.")


if __name__ == "__main__":
    main()