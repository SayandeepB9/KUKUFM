import json
import random
from typing import List, Dict, Any, Optional
import os
from langchain_core.prompts import ChatPromptTemplate
from llm_api import llm_api, get_model_from_config
from pydantic import BaseModel, Field

class PlotResponse(BaseModel):
    """Response model for plot generation."""
    
    detailed_plot: str = Field(
        description="A detailed plot narrative that connects the outline events into a cohesive story."
    )
    
    # Make literary_elements optional since we'll populate it after the LLM call
    literary_elements: Optional[Dict[str, str]] = Field(
        default=None,
        description="The literary elements incorporated into the plot."
    )
    
    class Config:
        # Explicitly set which fields are required to match OpenAI's expected format
        json_schema_extra = {
            "required": ["detailed_plot"]
        }

class PlotSelectorAgent:
    """Agent that develops detailed plots from outlines using literary elements."""
    
    def __init__(self, api_key=None):
        """Initialize the plot selector agent."""
        # Set model type for plot development
        self.model_type = "plot_development"
        
        # Get the model from config or use default
        self.model_name = get_model_from_config(self.model_type)
        
        # Initialize LLM using the llm_api function
        self.llm = llm_api(
            api_key=api_key,
            model_type=self.model_type,
            streaming=False
        )
        
        # Create structured output version of the LLM
        # Explicitly specify the function_calling method to avoid schema issues
        self.structured_llm = self.llm.with_structured_output(
            PlotResponse,
            method="function_calling"
        )
        
        # Load literary elements
        self.story_elements = self._load_story_elements()
        
        # Define prompt for plot development
        self.system_prompt = (
            "You are an expert storyteller specializing in developing compelling plots. "
            "Your task is to take a story outline (a sequence of key events) and expand it into a "
            "rich, detailed plot that connects these events in an engaging way. "
            "\n\nIncorporate the assigned literary elements seamlessly into your plot development. "
            "Show how characters evolve, how tension builds between events, and how each outline point "
            "connects to the next in a logical but surprising way."
            "\n\nCreate a plot that is specific, vivid, and internally consistent, while "
            "maintaining the core events from the original outline. Imagine the connective tissue "
            "between major events - what must happen to get from one point to the next."
            "\n\nYour response should be a detailed plot description (at least 500 words) that "
            "brings the outline to life through these literary elements, providing a roadmap "
            "for how the story will unfold."
        )
        
        self.human_prompt = (
            "STORY OUTLINE:\n{outline_str}\n\n"
            "SELECTED LITERARY ELEMENTS TO INCORPORATE:\n"
            "{literary_elements_str}\n\n"
            "Please develop a detailed plot that connects all the outline events while incorporating "
            "these randomly selected literary elements. Show how the story progresses logically from one event to the next, "
            "while building tension and character development throughout."
        )
        
        # Set up the prompt template
        self.plot_development_prompt = ChatPromptTemplate.from_messages([
            ("system", self.system_prompt),
            ("human", self.human_prompt)
        ])
        
        # Create the chain
        self.plot_development_chain = self.plot_development_prompt | self.structured_llm
    
    def _load_story_elements(self) -> Dict[str, List[str]]:
        """Load story elements from JSON file."""
        try:
            # Try to load from the current directory
            file_path = "story_elements.json"
            if os.path.exists(file_path):
                with open(file_path, 'r', encoding='utf-8') as file:
                    data = json.load(file)
                    return data.get("literary_elements", {})
        except Exception as e:
            print(f"Error loading story elements: {e}")
            # Return empty dictionary if file loading fails
            return {}
    
    def _select_random_elements(self) -> Dict[str, str]:
        """Randomly select which categories to include and a random element from each chosen category."""
        # Get all available categories
        categories = list(self.story_elements.keys())[:5]
        
        # Randomly decide how many categories to include (at least 1, at most all)
        num_categories = random.randint(3, len(categories))
        
        # Randomly select which categories to include
        selected_categories = random.sample(categories, num_categories)
        
        # For each selected category, choose a random element
        selected_elements = {}
        for category in selected_categories:
            if self.story_elements[category]:  # Only select if the category has elements
                selected_elements[category] = random.choice(self.story_elements[category])
        
        return selected_elements
    
    def generate_plot(self, outline: List[str]) -> PlotResponse:
        """Generate a detailed plot from the outline using random literary elements."""
        # Format the outline as a string
        outline_str = "\n".join([f"{i+1}. {event}" for i, event in enumerate(outline)])
        
        # Select random literary elements
        literary_elements = self._select_random_elements()
        
        # Print selected elements for reference
        print("\n=== RANDOMLY SELECTED LITERARY ELEMENTS ===")
        for category, element in literary_elements.items():
            print(f"{category.capitalize()}: {element}")
        
        # Format literary elements as a string for the prompt
        literary_elements_str = "\n".join([f"- {category.capitalize()}: {element}" for category, element in literary_elements.items()])
        
        # Prepare parameters for the prompt
        params = {
            "outline_str": outline_str,
            "literary_elements_str": literary_elements_str
        }
        
        try:
            print("\nGenerating detailed plot... (this may take a moment)\n")
            
            # Invoke the LLM to generate the plot
            response = self.plot_development_chain.invoke(params)
            
            # Create a new PlotResponse with both the generated plot and our literary elements
            result = PlotResponse(
                detailed_plot=response.detailed_plot,
                literary_elements=literary_elements
            )
            
            return result
            
        except Exception as e:
            print(f"Error generating plot: {e}")
            # Create a PlotResponse object with error information
            return PlotResponse(
                detailed_plot=f"Error generating plot. Please try again. Error: {str(e)}",
                literary_elements=literary_elements
            )
    
    def save_plot(self, plot_data: PlotResponse, filename: str = None) -> str:
        """Save the generated plot to a JSON file."""
        if not filename:
            import datetime
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"plot_{timestamp}.json"
        
        try:
            # Convert PlotResponse to dict for JSON serialization
            plot_dict = plot_data.model_dump()
            
            with open(filename, 'w', encoding='utf-8') as file:
                json.dump(plot_dict, file, indent=2, ensure_ascii=False)
            print(f"\nPlot saved to {filename}")
            return filename
        except Exception as e:
            print(f"Error saving plot: {e}")
            return None

def main():
    """Main function to run the plot selector agent."""
    # Display header
    print("\n=== PLOT SELECTOR AGENT ===\n")
    
    # Get API key from environment or config
    api_key = os.environ.get("GROQ_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("Please enter your API key (Groq or OpenAI):")
        api_key = input("> ")
    
    try:
        # Initialize the plot selector agent
        plot_agent = PlotSelectorAgent(api_key=api_key)
        
        # Sample outline for testing
        sample_outline = [
            'Guests arrive at the hotel for a mysterious conference',
            'The guests experience strange occurrences in their rooms',
            "The hotel's dark past is revealed through a series of eerie hints",
            'The guests begin to disappear one by one',
            'The main character discovers a hidden room with a terrifying secret',
            "The hotel's ghostly presence becomes increasingly aggressive",
            'The main character must escape the hotel alive'
        ]
        
        # Ask if user wants to use sample outline or input their own
        print("Would you like to use the sample outline about the mysterious hotel? (y/n)")
        use_sample = input("> ").lower().startswith('y')
        
        if use_sample:
            outline = sample_outline
            print("\nUsing sample outline:")
            for i, event in enumerate(outline, 1):
                print(f"{i}. {event}")
        else:
            # Get outline from user
            print("\nPlease enter your story outline events, one per line.")
            print("Enter an empty line when finished.")
            
            outline = []
            i = 1
            while True:
                event = input(f"{i}. ")
                if not event:
                    break
                outline.append(event)
                i += 1
        
        if not outline:
            print("No outline provided. Using sample outline.")
            outline = sample_outline
        
        # Generate plot
        plot_result = plot_agent.generate_plot(outline)
        
        # Display the detailed plot
        print("\n=== GENERATED PLOT ===\n")
        print(plot_result.detailed_plot)
        
        # Ask if user wants to save the plot
        print("\nWould you like to save this plot? (y/n)")
        save_plot = input("> ").lower().startswith('y')
        
        if save_plot:
            plot_agent.save_plot(plot_result)
        
    except Exception as e:
        print(f"\nAn unexpected error occurred: {str(e)}")
        print("Please check your API key and internet connection, then try again.")

if __name__ == "__main__":
    main()