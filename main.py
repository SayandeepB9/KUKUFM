import os
from dotenv import load_dotenv
import argparse
import yaml
import json
from datetime import datetime

# Load environment variables
load_dotenv()

# Import local components
from outline_generation_agent import OutlineGenerator
from character_development_agent import CharacterDevelopmentAgent
from plot_selector import StoryElementLibrary
from splitter_agent import StorySplitterAgent, Episode
from llm_api import llm_api

def load_config():
    """Load configuration from config.yaml"""
    try:
        with open('config.yaml', 'r') as file:
            return yaml.safe_load(file)
    except Exception as e:
        print(f"Error loading config: {e}")
        return None

# Custom JSON encoder to handle Episode objects
class StoryJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, Episode):
            return {
                "number": obj.number,
                "title": obj.title,
                "content": obj.content,
                "cliffhanger": obj.cliffhanger
            }
        # Let the base class handle other types or raise TypeError
        return json.JSONEncoder.default(self, obj)

def save_story(story_data, output_dir="generated_stories"):
    """Save generated story to JSON and text files"""
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    # Create timestamp for unique filenames
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Sanitize story title for filename
    title_slug = "".join(c if c.isalnum() else "_" for c in story_data["topic"][:30])
    base_filename = f"{output_dir}/{timestamp}_{title_slug}"
    
    # Save complete story data as JSON using custom encoder
    with open(f"{base_filename}.json", 'w', encoding='utf-8') as f:
        json.dump(story_data, f, indent=2, ensure_ascii=False, cls=StoryJSONEncoder)
        
    # Save readable story text
    with open(f"{base_filename}.txt", 'w', encoding='utf-8') as f:
        f.write(f"# {story_data['topic']}\n\n")
        
        f.write("## Story Outline\n")
        for i, event in enumerate(story_data['outline'], 1):
            f.write(f"{i}. {event}\n")
        f.write("\n")
        
        f.write("## Characters\n")
        for char in story_data['characters']:
            f.write(f"### {char['name']} ({char['role']})\n")
            f.write(f"{char['description']}\n\n")
            
        f.write("## Episodes\n")
        for episode in story_data['episodes']:
            f.write(f"### Episode {episode.number}: {episode.title}\n\n")
            f.write(f"{episode.content}\n\n")
            if episode.cliffhanger:
                f.write(f"**Cliffhanger:** {episode.cliffhanger}\n\n")
    
    print(f"\nStory saved to {base_filename}.json and {base_filename}.txt")
    return base_filename


def generate_story_pipeline(topic, num_episodes=5, story_type="general"):
    """Run the complete story generation pipeline"""
    print(f"\n=== Generating story for topic: '{topic}' ===\n")
    
    # Step 1: Generate outline
    print("Step 1/4: Generating story outline...")
    outline_generator = OutlineGenerator()
    outline = outline_generator.generate_outline(topic)
    
    # Convert outline to a single string for downstream use
    outline_text = " ".join(outline)
    
    # Step 2: Generate characters
    print("\nStep 2/4: Developing characters...")
    character_agent = CharacterDevelopmentAgent()
    characters = character_agent.generate_characters(outline_text)
    
    # Step 3: Select plot elements
    print("\nStep 3/4: Generating plot elements...")
    plot_library = StoryElementLibrary()
    plot_options = plot_library.generate_plot_options(outline_text, story_type)
    
    # Auto-select 3-5 plot points based on the number of episodes
    num_plots = min(max(3, num_episodes - 2), 5)
    selected_plots = plot_options[:num_plots]
    
    print(f"\nSelected {len(selected_plots)} plot elements automatically:")
    for i, plot in enumerate(selected_plots, 1):
        print(f"{i}. {plot}")
    
    # Step 4: Split into episodes
    print("\nStep 4/4: Splitting story into episodes...")
    splitter = StorySplitterAgent()
    episodes = splitter.split_story(outline_text, selected_plots, characters, num_episodes=num_episodes)
    
    # Create story data structure
    story_data = {
        "topic": topic,
        "story_type": story_type,
        "outline": outline,
        "characters": characters,
        "plots": selected_plots,
        "episodes": episodes,  # this will now be properly serialized with our custom encoder
        "generated_at": datetime.now().isoformat()
    }
    
    # Save the story
    save_story(story_data)
    
    print("\n=== Story generation complete! ===")
    print(f"Generated {len(episodes)} episodes with {len(characters)} characters")
    
    return story_data


def main():
    """Main function to run the story generation pipeline"""
    parser = argparse.ArgumentParser(description="Generate a story using AI")
    parser.add_argument("topic", help="Topic or theme for the story")
    parser.add_argument("--episodes", type=int, default=5, help="Number of episodes (default: 5)")
    parser.add_argument("--type", default="general", help="Story type (e.g., mystery, sci-fi, fantasy)")
    
    args = parser.parse_args()
    
    # Load and validate configuration
    config = load_config()
    if not config:
        print("Failed to load configuration. Make sure config.yaml exists.")
        return
    
    # Check for API keys
    api_key_env = os.getenv('GROQ_API_KEY') or os.getenv('OPENAI_API_KEY')
    api_key_config = config['api_keys'].get('groq') or config['api_keys'].get('openai')
    
    if not api_key_env and not api_key_config:
        print("ERROR: No API key found. Please set GROQ_API_KEY or OPENAI_API_KEY in .env or config.yaml")
        return
    
    # Generate the story
    try:
        generate_story_pipeline(args.topic, args.episodes, args.type)
    except Exception as e:
        print(f"Error generating story: {e}")

if __name__ == "__main__":
    main()
