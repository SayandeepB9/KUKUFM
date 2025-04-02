from outline_generation_agent import OutlineGenerator
from plot_selector import StoryElementLibrary, extract_outline_from_input
from consistency_checker import integrate_consistency_checker, ConsistencyIssue
import os
import json
import datetime
import re
from typing import List, Tuple

def main():
    # Display header
    current_datetime = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    current_user = os.getenv("USER", "Unknown")
    
    print(f"Current Date and Time: {current_datetime}")
    print(f"Current User: {current_user}")
    print("\n=== AI STORY GENERATION PIPELINE WITH CONSISTENCY CHECKING ===\n")
    
    # Get API key from environment or config
    api_key = os.environ.get("GROQ_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("Please enter your API key (Groq or OpenAI):")
        api_key = input("> ")
    
    try:
        # Initialize components
        outline_generator = OutlineGenerator(api_key=api_key)
        plot_library = StoryElementLibrary(api_key=api_key)
        consistency_checker = integrate_consistency_checker(api_key=api_key)
        
        # Step 1: Get story topic
        print("\nPlease enter a topic for your story:")
        topic = input("> ")
        
        # Step 2: Generate story outline
        print("\nGenerating story outline...")
        outline_events = outline_generator.generate_outline(topic)
        
        # Step 3: Determine story type
        print("\nPlease specify the story type (e.g., ghost, sci-fi, fantasy, mystery, etc.):")
        story_type = input("> ").lower()
        
        # Step 4: Generate plot options
        plot_options = plot_library.generate_plot_options(outline_events, story_type)
        
        # Step 5: Display plot options
        plot_library.display_plot_options(plot_options)
        
        # Step 6: Ask if user wants to check consistency
        print("\nWould you like to check these plot options for consistency? (y/n)")
        if input("> ").lower().startswith('y'):
            # Run consistency check
            issues = consistency_checker.check_plot_consistency(story_type, outline_events, plot_options)
            consistency_ok = consistency_checker.display_consistency_report(issues)
            
            # If there are issues, ask if user wants to regenerate problematic options
            if issues:
                print("\nWould you like to regenerate plot options with improved consistency? (y/n)")
                if input("> ").lower().startswith('y'):
                    # Identify problematic plot options
                    problem_indices = sorted(set(issue.plot_option_index for issue in issues if issue.plot_option_index is not None))
                    problem_plots = [(idx, plot_options[idx]) for idx in problem_indices if 0 <= idx < len(plot_options)]
                    
                    if problem_plots:
                        print("\nGenerating improved plot options...")
                        improved_options = consistency_checker.generate_improved_suggestions(
                            story_type, outline_events, problem_plots, issues
                        )
                        
                        # Replace the problematic options with improved versions
                        for i, (idx, _) in enumerate(problem_plots):
                            if i < len(improved_options) and 0 <= idx < len(plot_options):
                                plot_options[idx] = improved_options[i]
                        
                        # Show the updated options
                        print("\n=== UPDATED PLOT OPTIONS ===")
                        plot_library.display_plot_options(plot_options)
                        
                        # Run consistency check again to verify improvements
                        print("\nChecking consistency of updated options...")
                        new_issues = consistency_checker.check_plot_consistency(story_type, outline_events, plot_options)
                        consistency_checker.display_consistency_report(new_issues)
        
        # Step 7: Select plot options
        selected_options = plot_library.select_plot_options(plot_options)
        
        # Save selected options
        plot_library.save_selected_options(selected_options)
        
        print("\nPlot selection complete! You can now proceed to the next stage of your pipeline.")
        
    except Exception as e:
        print(f"\nAn unexpected error occurred: {str(e)}")
        print("Please check your API key and internet connection, then try again.")

if __name__ == "__main__":
    main()