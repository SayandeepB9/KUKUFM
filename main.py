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
from translator_agent import TranslatorAgent
from llm_api import llm_api

def load_config():
    """Load configuration from config.yaml"""
    try:
        with open('config.yaml', 'r') as file:
            return yaml.safe_load(file)
    except Exception as e:
        print(f"Error loading config: {e}")
        return None

# Custom JSON encoder to handle Episode objects and LengthenedEpisode objects
class StoryJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, Episode):
            return {
                "number": obj.number,
                "title": obj.title,
                "content": obj.content,
                "cliffhanger": obj.cliffhanger
            }
        # Handle LengthenedEpisode objects
        elif hasattr(obj, 'lengthened_content'):
            return {
                "lengthened_content": obj.lengthened_content,
                "engagement_points": getattr(obj, 'engagement_points', []),
                "summary": getattr(obj, 'summary', "")
            }
        # Let the base class handle other types or raise TypeError
        return json.JSONEncoder.default(self, obj)

def create_story_directory(topic):
    """Create a unique directory for this story based on name and timestamp"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # Sanitize story title for directory name
    title_slug = "".join(c if c.isalnum() else "_" for c in topic[:30])
    story_dir = f"stories/{title_slug}_{timestamp}"
    
    # Create directories
    if not os.path.exists(story_dir):
        os.makedirs(story_dir)
    
    # Create subdirectories for different outputs
    os.makedirs(f"{story_dir}/episodes", exist_ok=True)
    os.makedirs(f"{story_dir}/dialogue", exist_ok=True)
    
    return story_dir

def save_story(story_data, story_dir):
    """Save generated story to JSON and text files in the story directory"""
    # Save complete story data as JSON using custom encoder
    with open(f"{story_dir}/story_data.json", 'w', encoding='utf-8') as f:
        json.dump(story_data, f, indent=2, ensure_ascii=False, cls=StoryJSONEncoder)
        
    # Save readable story text
    with open(f"{story_dir}/story_details.md", 'w', encoding='utf-8') as f:
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
    
    print(f"\nStory data saved to {story_dir}/story_data.json and {story_dir}/story_details.md")
    return story_dir

def save_final_story(story_data, story_dir):
    """Save the final story with dialogues to a single file for publishing"""
    # Write the final story to a file
    filename = f"{story_dir}/final_story.md"
    
    with open(filename, 'w', encoding='utf-8') as f:
        # Title and intro
        f.write(f"# {story_data['topic']}\n\n")
        
        # Introduction to the story - combine the outline points
        f.write("## Introduction\n\n")
        outline_text = " ".join(story_data['outline'])
        f.write(f"{outline_text}\n\n")
        
        # Characters introduction
        f.write("## Characters\n\n")
        for char in story_data['characters']:
            f.write(f"**{char['name']}** ({char['role']}): {char['description']}\n\n")
        
        # Episodes with dialogues
        f.write("## Story\n\n")
        
        for episode in story_data['episodes']:
            episode_num = episode.number
            
            # Get dialogue for this episode
            dialogue = story_data.get('dialogue', {}).get(episode_num)
            
            if dialogue:
                f.write(f"### Episode {episode_num}: {episode.title}\n\n")
                f.write(f"{dialogue}\n\n")
                
                # Add a separator between episodes
                if episode_num < len(story_data['episodes']):
                    f.write("---\n\n")
        
        # Add metadata at the end
        f.write(f"\n\n*Generated on {datetime.now().strftime('%Y-%m-%d')}*\n")
    
    print(f"\nFinal story saved to {filename}")
    return filename

def handle_outline_feedback(outline_generator, topic, outline):
    """Handle human feedback for refining the story outline"""
    while True:
        satisfaction = input("\nAre you satisfied with this outline? (yes/no): ").strip().lower()
        
        if satisfaction in ['y', 'yes']:
            print("Great! Proceeding with the current outline.")
            return outline
        elif satisfaction in ['n', 'no']:
            feedback = input("\nPlease provide feedback on how to improve the outline: ")
            print(f"\nRefining outline based on feedback: '{feedback}'")
            
            # Use the refine_outline method to improve the outline
            refined_outline = outline_generator.refine_outline(topic, outline, feedback)
            
            # Show the refined outline to the user
            print("\nRefined outline:")
            for i, event in enumerate(refined_outline, 1):
                print(f"{i}. {event}")
                
            # Update the outline
            outline = refined_outline
        else:
            print("Please enter 'yes' or 'no'.")
            continue

def translate_story(story_file, target_language, story_dir):
    """Translate the final story to the target language"""
    print(f"\nTranslating story to {target_language}...")
    
    # Read the content of the final story
    with open(story_file, 'r', encoding='utf-8') as f:
        story_content = f.read()
    
    # Initialize translator agent
    translator = TranslatorAgent()
    
    # Translate the story
    translated_content = translator.translate_story(story_content, target_language)
    
    # Save the translated content to a new file
    translated_file = os.path.join(story_dir, f"final_story_{target_language.lower()}.md")
    with open(translated_file, 'w', encoding='utf-8') as f:
        f.write(translated_content)
    
    print(f"Translated story saved to: {translated_file}")
    return translated_file

def generate_story_pipeline(topic, num_episodes=5, story_type="general", target_language=None):
    """Run the complete story generation pipeline"""
    print(f"\n=== Generating story for topic: '{topic}' ===\n")
    
    # Create a unique directory for this story
    story_dir = create_story_directory(topic)
    
    # Step 1: Generate outline
    print("Step 1/6: Generating story outline...")
    outline_generator = OutlineGenerator()
    outline = outline_generator.generate_outline(topic)
    
    # Add human feedback loop for outline refinement
    outline = handle_outline_feedback(outline_generator, topic, outline)
    
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
    
    previous_episodes_summary = ""
    previous_cliffhanger = ""
    
    for i, episode in enumerate(episodes):
        print(f"Enhancing episode {episode.number}: {episode.title}...")
        
        # Process each episode
        enhanced = lengthener.lengthen_episode(
            episode_title=episode.title,
            episode_number=episode.number,
            episode_outline=episode.content,
            previous_episodes_summary=previous_episodes_summary,
            previous_cliffhanger=previous_cliffhanger,
            include_cliffhanger=bool(episode.cliffhanger)
        )
        
        # Store the enhanced episode
        enhanced_episodes[episode.number] = enhanced
        
        # Update previous episode summary for next iteration
        if hasattr(enhanced, 'summary'):
            previous_episodes_summary += enhanced.summary + " "
        else:
            previous_episodes_summary += f"Episode {episode.number}: {episode.title}. "
        
        previous_cliffhanger = episode.cliffhanger if episode.cliffhanger else ""
        
        # Save to file within the story directory
        file_name = f"Episode_{episode.number}_{episode.title.replace(' ', '_')}.md"
        output_path = os.path.join(story_dir, "episodes", file_name)
        lengthener.save_episode_to_file(enhanced, output_path)
    
    # Step 6: Generate dialogue
    print("\nStep 6/6: Generating dialogue for episodes...")
    dialogue_agent = DialogueAgent()
    dialogues = {}
    
    # Process all episodes for dialogue generation
    for episode in episodes:
        print(f"Generating dialogue for episode {episode.number}: {episode.title}...")
        
        # Use enhanced content if available
        enhanced_episode = enhanced_episodes.get(episode.number)
        if enhanced_episode and hasattr(enhanced_episode, 'lengthened_content'):
            print("Using enhanced content for dialogue generation...")
            episode_content = enhanced_episode.lengthened_content
        else:
            episode_content = episode.content
        
        # Generate dialogue
        dialogue = dialogue_agent.generate_dialogue(
            story_type=story_type,
            storyline=episode_content,
            characters=characters
        )
        
        # Store the dialogue and save to separate file 
        dialogues[episode.number] = dialogue
        
        # Save dialogue to separate file
        dialogue_file = os.path.join(story_dir, "dialogue", f"dialogue_episode_{episode.number}.md")
        with open(dialogue_file, 'w', encoding='utf-8') as f:
            f.write(f"# Dialogue for Episode {episode.number}: {episode.title}\n\n")
            f.write(dialogue)
    
    # Create story data structure - convert enhanced_episodes to serializable format
    serializable_enhanced_episodes = {}
    for num, ep in enhanced_episodes.items():
        if hasattr(ep, 'lengthened_content'):
            serializable_enhanced_episodes[num] = {
                "lengthened_content": ep.lengthened_content,
                "engagement_points": getattr(ep, 'engagement_points', []),
                "summary": getattr(ep, 'summary', "")
            }
        else:
            serializable_enhanced_episodes[num] = ep
    
    story_data = {
        "topic": topic,
        "story_type": story_type,
        "outline": outline,
        "characters": characters,
        "plots": selected_plots,
        "episodes": episodes,
        "enhanced_episodes": serializable_enhanced_episodes,
        "dialogue": dialogues,
        "generated_at": datetime.now().isoformat()
    }
    
    # Save all story data to the story directory
    save_story(story_data, story_dir)
    
    # Save the final publishable story
    final_story_file = save_final_story(story_data, story_dir)
    
    # Step 7 (Optional): Translate the final story if a target language is specified
    translated_file = None
    if target_language:
        translated_file = translate_story(final_story_file, target_language, story_dir)
    
    print("\n=== Story generation complete! ===")
    print(f"Generated {len(episodes)} episodes with {len(characters)} characters")
    print(f"Enhanced {len(enhanced_episodes)} episodes with detailed content")
    print(f"Generated dialogue for {len(dialogues)} episodes")
    print(f"All story files saved to directory: {story_dir}")
    
    if translated_file:
        print(f"Story translated to {target_language} and saved to: {translated_file}")
    
    return story_data, story_dir

def main():
    """Main function to run the story generation pipeline"""
    parser = argparse.ArgumentParser(description="Generate a story using AI")
    parser.add_argument("topic", help="Topic or theme for the story")
    parser.add_argument("--episodes", type=int, default=5, help="Number of episodes (default: 5)")
    parser.add_argument("--type", default="general", help="Story type (e.g., mystery, sci-fi, fantasy)")
    parser.add_argument("--translate", help="Translate the final story to this language (e.g., Hindi, French, Spanish)")
    
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
        story_data, story_dir = generate_story_pipeline(
            args.topic, 
            args.episodes, 
            args.type, 
            args.translate
        )
        # Return both story data and directory for potential further processing
    except Exception as e:
        print(f"Error generating story: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
