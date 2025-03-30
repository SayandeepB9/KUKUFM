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
from enhancement import EpisodeLengtheningAgent
from dialogue_generation_agent import DialogueAgent
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
            
            # Use enhanced content if available
            content = story_data.get('enhanced_episodes', {}).get(
                episode.number, {}).get('lengthened_content', episode.content)
            f.write(f"{content}\n\n")
            
            # Add dialogue if available
            dialogue = story_data.get('dialogue', {}).get(episode.number)
            if dialogue:
                f.write("## Dialogue\n")
                f.write(f"{dialogue}\n\n")
            
            if episode.cliffhanger:
                f.write(f"**Cliffhanger:** {episode.cliffhanger}\n\n")
    
    print(f"\nStory saved to {base_filename}.json and {base_filename}.txt")
    return base_filename


def generate_story_pipeline(topic, num_episodes=5, story_type="general"):
    """Run the complete story generation pipeline"""
    print(f"\n=== Generating story for topic: '{topic}' ===\n")
    
    # Step 1: Generate outline
    print("Step 1/6: Generating story outline...")
    outline_generator = OutlineGenerator()
    outline = outline_generator.generate_outline(topic)
    
    # Convert outline to a single string for downstream use
    outline_text = " ".join(outline)
    
    # Step 2: Generate characters
    print("\nStep 2/6: Developing characters...")
    character_agent = CharacterDevelopmentAgent()
    characters = character_agent.generate_characters(outline_text)
    
    # Step 3: Select plot elements
    print("\nStep 3/6: Generating plot elements...")
    plot_library = StoryElementLibrary()
    plot_options = plot_library.generate_plot_options(outline_text, story_type)
    
    # Auto-select 3-5 plot points based on the number of episodes
    num_plots = min(max(3, num_episodes - 2), 5)
    selected_plots = plot_options[:num_plots]
    
    print(f"\nSelected {len(selected_plots)} plot elements automatically:")
    for i, plot in enumerate(selected_plots, 1):
        print(f"{i}. {plot}")
    
    # Step 4: Split into episodes
    print("\nStep 4/6: Splitting story into episodes...")
    splitter = StorySplitterAgent()
    episodes = splitter.split_story(outline_text, selected_plots, characters, num_episodes=num_episodes)
    
    # Step 5: Enhance episodes with lengthening
    print("\nStep 5/6: Enhancing episodes with detailed content...")
    lengthener = EpisodeLengtheningAgent()
    enhanced_episodes = {}
    
    # Process only the first episode for demonstration purposes (to keep runtime reasonable)
    # In production, you might want to process all or a subset of episodes
    if episodes and len(episodes) > 0:
        first_episode = episodes[0]
        print(f"Enhancing episode {first_episode.number}: {first_episode.title}...")
        
        previous_episodes_summary = ""  # First episode has no previous episodes
        previous_cliffhanger = ""
        
        # Process the first episode
        enhanced = lengthener.lengthen_episode(
            episode_title=first_episode.title,
            episode_number=first_episode.number,
            episode_outline=first_episode.content,
            previous_episodes_summary=previous_episodes_summary,
            previous_cliffhanger=previous_cliffhanger,
            include_cliffhanger=bool(first_episode.cliffhanger)
        )
        
        # Store the enhanced episode
        enhanced_episodes[first_episode.number] = enhanced
        
        # Save to file
        output_dir = "output_story"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        file_name = f"Episode_{first_episode.number}_{first_episode.title.replace(' ', '_')}.md"
        output_path = os.path.join(output_dir, file_name)
        lengthener.save_episode_to_file(enhanced, output_path)
    
    # Step 6: Generate dialogue
    print("\nStep 6/6: Generating dialogue for episodes...")
    dialogue_agent = DialogueAgent()
    dialogues = {}
    
    # Again, just process the first episode for demonstration
    if episodes and len(episodes) > 0:
        first_episode = episodes[0]
        print(f"Generating dialogue for episode {first_episode.number}: {first_episode.title}...")
        
        # Generate dialogue
        dialogue = dialogue_agent.generate_dialogue(
            story_type=story_type,
            storyline=first_episode.content,
            characters=characters
        )
        
        # Store the dialogue
        dialogues[first_episode.number] = dialogue
    
    # Create story data structure
    story_data = {
        "topic": topic,
        "story_type": story_type,
        "outline": outline,
        "characters": characters,
        "plots": selected_plots,
        "episodes": episodes,
        "enhanced_episodes": enhanced_episodes,
        "dialogue": dialogues,
        "generated_at": datetime.now().isoformat()
    }
    
    # Save the story
    save_story(story_data)
    
    print("\n=== Story generation complete! ===")
    print(f"Generated {len(episodes)} episodes with {len(characters)} characters")
    print(f"Enhanced {len(enhanced_episodes)} episodes with detailed content")
    print(f"Generated dialogue for {len(dialogues)} episodes")
    
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
