import os
from dotenv import load_dotenv
import argparse
import yaml
import json
from datetime import datetime
import concurrent.futures
from tqdm import tqdm  # For progress bars

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

def handle_character_feedback(character_agent, plot, characters):
    """Handle human feedback for refining the story characters"""
    while True:
        satisfaction = input("\nAre you satisfied with these characters? (yes/no): ").strip().lower()
        
        if satisfaction in ['y', 'yes']:
            print("Great! Proceeding with the current characters.")
            return characters
        elif satisfaction in ['n', 'no']:
            feedback = input("\nPlease provide feedback on how to improve the characters: ")
            print(f"\nRefining characters based on feedback: '{feedback}'")
            
            # Use the refine_characters method to improve the characters
            refined_characters = character_agent.refine_characters(plot, characters, feedback)
            
            # Show the refined characters to the user
            print("\nRefined characters:")
            for character in refined_characters:
                print(f"Name: {character['name']}")
                print(f"Description: {character['description']}")
                print(f"Role: {character['role']}")
                print("------")
                
            # Update the characters
            characters = refined_characters
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

def translate_story_parallel(story_file, target_languages, story_dir):
    """Translate the final story to multiple target languages in parallel"""
    if not target_languages:
        return []
        
    print(f"\nTranslating story to {len(target_languages)} languages in parallel...")
    
    # Read the content of the final story once
    with open(story_file, 'r', encoding='utf-8') as f:
        story_content = f.read()
    
    # Initialize translator agent
    translator = TranslatorAgent()
    translated_files = []
    
    # Function to translate to a single language
    def translate_to_language(language):
        try:
            # Translate the story
            translated_content = translator.translate_story(story_content, language)
            
            # Save the translated content to a new file
            translated_file = os.path.join(story_dir, f"final_story_{language.lower()}.md")
            with open(translated_file, 'w', encoding='utf-8') as f:
                f.write(translated_content)
            
            return language, translated_file
        except Exception as e:
            print(f"Error translating to {language}: {e}")
            return language, None
    
    # Process translations in parallel
    max_workers = min(10, len(target_languages))  # Limit concurrent API calls
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all translation tasks
        future_to_language = {executor.submit(translate_to_language, lang): lang for lang in target_languages}
        
        # Process results as they complete
        for future in tqdm(concurrent.futures.as_completed(future_to_language), total=len(target_languages), desc="Translating"):
            try:
                language, file_path = future.result()
                if file_path:
                    translated_files.append((language, file_path))
                    print(f"✓ Translated to {language}: {file_path}")
            except Exception as e:
                language = future_to_language[future]
                print(f"× Failed to translate to {language}: {e}")
    
    return translated_files

def translate_episodes_parallel(episodes, enhanced_episodes, dialogues, target_language, story_dir):
    """
    Translate individual episodes in parallel for a single target language.
    
    Args:
        episodes: List of Episode objects
        enhanced_episodes: Dict of enhanced episode content
        dialogues: Dict of episode dialogues
        target_language: Target language for translation
        story_dir: Story directory to save translations
        
    Returns:
        Dict containing translation results for each episode
    """
    print(f"\nTranslating {len(episodes)} episodes to {target_language} in parallel...")
    
    # Create a subdirectory for translated episodes
    translated_dir = os.path.join(story_dir, f"translated_{target_language.lower()}")
    os.makedirs(translated_dir, exist_ok=True)
    
    # Initialize translator agent
    translator = TranslatorAgent()
    
    # Results container
    translated_episodes = {}
    
    # Function to translate a single episode
    def translate_episode(episode):
        try:
            # Get enhanced content if available
            enhanced_content = enhanced_episodes.get(episode.number, {}).get('lengthened_content', episode.content)
            dialogue_content = dialogues.get(episode.number, "")
            
            # Combine content for translation
            episode_content = (
                f"# Episode {episode.number}: {episode.title}\n\n"
                f"{enhanced_content}\n\n"
                f"## Dialogue\n\n{dialogue_content}"
            )
            
            # Translate the content
            translated_content = translator.translate_story(episode_content, target_language)
            
            # Save to file
            output_file = os.path.join(
                translated_dir, 
                f"episode_{episode.number}_{target_language.lower()}.md"
            )
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(translated_content)
            
            return episode.number, {
                "translated_content": translated_content,
                "file_path": output_file
            }
        except Exception as e:
            print(f"Error translating episode {episode.number}: {e}")
            return episode.number, {"error": str(e)}
    
    # Process translations in parallel
    max_workers = min(10, len(episodes))  # Limit concurrent API calls
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all translation tasks
        future_to_episode = {executor.submit(translate_episode, episode): episode for episode in episodes}
        
        # Process results as they complete
        for future in tqdm(concurrent.futures.as_completed(future_to_episode), 
                           total=len(episodes), 
                           desc=f"Translating episodes to {target_language}"):
            try:
                episode_number, result = future.result()
                translated_episodes[episode_number] = result
                if "error" not in result:
                    print(f"✓ Episode {episode_number} translated to {target_language}")
                else:
                    print(f"× Failed to translate episode {episode_number}: {result['error']}")
            except Exception as e:
                episode = future_to_episode[future]
                print(f"× Error processing episode {episode.number}: {e}")
    
    # Create a combined translated file with all episodes
    combined_file = os.path.join(story_dir, f"full_story_{target_language.lower()}.md")
    
    try:
        with open(combined_file, 'w', encoding='utf-8') as f:
            for episode in sorted(episodes, key=lambda e: e.number):
                translation = translated_episodes.get(episode.number, {}).get('translated_content')
                if translation:
                    f.write(f"{translation}\n\n---\n\n")
        
        print(f"Combined translated story saved to: {combined_file}")
    except Exception as e:
        print(f"Error creating combined translation file: {e}")
    
    return translated_episodes

def generate_story_pipeline(topic, num_episodes=5, story_type="general", target_languages=None):
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
    
    # Add human feedback loop for character refinement
    characters = handle_character_feedback(character_agent, outline_text, characters)
    
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
    
    # Step 5: Enhance episodes with lengthening - using parallel processing
    print("\nStep 5/6: Enhancing episodes with detailed content in parallel...")
    lengthener = EpisodeLengtheningAgent()
    enhanced_episodes = {}
    
    # Pre-compute episode contexts
    episode_contexts = []
    previous_episodes_summary = ""
    previous_cliffhanger = ""
    
    for i, episode in enumerate(episodes):
        # Generate future episodes outlines to provide context
        future_episodes_outlines = ""
        if i < len(episodes) - 1:
            # Include up to 3 future episodes or whatever is available
            future_episodes = episodes[i+1:i+4]
            future_outlines = []
            for j, future_ep in enumerate(future_episodes):
                future_outlines.append(f"Episode {future_ep.number} - {future_ep.title}: {future_ep.content}")
            future_episodes_outlines = "\n\n".join(future_outlines)
        
        context = {
            "episode": episode,
            "episode_title": episode.title,
            "episode_number": episode.number,
            "episode_outline": episode.content,
            "previous_episodes_summary": previous_episodes_summary,
            "previous_cliffhanger": previous_cliffhanger,
            "include_cliffhanger": bool(episode.cliffhanger),
            "future_episodes_outlines": future_episodes_outlines,
            "characters": characters  # Pass the character data to the enhancement context
        }
        episode_contexts.append(context)
        
        # Update for next iteration
        previous_episodes_summary += str(episode.number) + ". " + episode.title + "\n" + episode.content + "\n\n"
        previous_cliffhanger = episode.cliffhanger if episode.cliffhanger else ""
    
    # Function to enhance a single episode
    def enhance_episode(context):
        episode = context["episode"]
        enhanced = lengthener.lengthen_episode(
            episode_title=context["episode_title"],
            episode_number=context["episode_number"],
            episode_outline=context["episode_outline"],
            previous_episodes_summary=context["previous_episodes_summary"],
            previous_cliffhanger=context["previous_cliffhanger"],
            include_cliffhanger=context["include_cliffhanger"],
            future_episodes_outlines=context["future_episodes_outlines"],
            characters=context["characters"]  # Pass character data to the lengthening function
        )
        
        # Save to file within the story directory
        file_name = f"Episode_{episode.number}_{episode.title.replace(' ', '_')}.md"
        output_path = os.path.join(story_dir, "episodes", file_name)
        lengthener.save_episode_to_file(enhanced, output_path)
        
        return episode.number, enhanced
    
    # Process episodes in parallel
    max_workers = min(10, len(episodes))  # Limit the number of concurrent API calls
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_episode = {executor.submit(enhance_episode, context): context for context in episode_contexts}
        
        # Process results as they complete
        for future in tqdm(concurrent.futures.as_completed(future_to_episode), total=len(episodes), desc="Enhancing episodes"):
            try:
                episode_number, enhanced = future.result()
                enhanced_episodes[episode_number] = enhanced
            except Exception as e:
                context = future_to_episode[future]
                print(f"Error enhancing episode {context['episode_number']}: {e}")
    
    # Step 6: Generate dialogue in parallel
    print("\nStep 6/6: Generating dialogue for episodes in parallel...")
    dialogue_agent = DialogueAgent()
    dialogues = {}
    
    # Function to generate dialogue for a single episode
    def generate_episode_dialogue(episode):
        # Use enhanced content if available
        enhanced_episode = enhanced_episodes.get(episode.number)
        if enhanced_episode and hasattr(enhanced_episode, 'lengthened_content'):
            episode_content = enhanced_episode.lengthened_content
        else:
            episode_content = episode.content
        
        # Generate dialogue
        dialogue = dialogue_agent.generate_dialogue(
            story_type=story_type,
            storyline=episode_content,
            characters=characters
        )
        
        # Save dialogue to separate file
        dialogue_file = os.path.join(story_dir, "dialogue", f"dialogue_episode_{episode.number}.md")
        with open(dialogue_file, 'w', encoding='utf-8') as f:
            f.write(f"# Dialogue for Episode {episode.number}: {episode.title}\n\n")
            f.write(dialogue)
        
        return episode.number, dialogue
    
    # Process dialogues in parallel
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_dialogue = {executor.submit(generate_episode_dialogue, episode): episode for episode in episodes}
        
        # Process results as they complete
        for future in tqdm(concurrent.futures.as_completed(future_to_dialogue), total=len(episodes), desc="Generating dialogues"):
            try:
                episode_number, dialogue = future.result()
                dialogues[episode_number] = dialogue
            except Exception as e:
                episode = future_to_dialogue[future]
                print(f"Error generating dialogue for episode {episode.number}: {e}")
    
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
    
    # Step 7 (Optional): Translation handling
    translated_files = []
    
    # Handle episode-by-episode translation for a single language
    if target_languages and isinstance(target_languages, (list, tuple)) and len(target_languages) == 1:
        target_language = target_languages[0]
        print(f"\nStep 7/7: Translating episodes to {target_language} in parallel...")
        translated_episodes = translate_episodes_parallel(
            episodes, 
            serializable_enhanced_episodes, 
            dialogues, 
            target_language, 
            story_dir
        )
        if translated_episodes:
            story_data["translations"] = {target_language: translated_episodes}
            # Update the JSON file with translation data
            with open(f"{story_dir}/story_data.json", 'w', encoding='utf-8') as f:
                json.dump(story_data, f, indent=2, ensure_ascii=False, cls=StoryJSONEncoder)
    
    # Handle full story translation for multiple languages
    elif target_languages and len(target_languages) > 1:
        print(f"\nStep 7/7: Translating full story to multiple languages...")
        translated_files = translate_story_parallel(final_story_file, target_languages, story_dir)
    
    # Handle single language as string (backward compatibility)
    elif target_languages and isinstance(target_languages, str):
        print(f"\nStep 7/7: Translating full story to {target_languages}...")
        translated_file = translate_story(final_story_file, target_languages, story_dir)
        if translated_file:
            translated_files.append((target_languages, translated_file))
    
    print("\n=== Story generation complete! ===")
    print(f"Generated {len(episodes)} episodes with {len(characters)} characters")
    print(f"Enhanced {len(enhanced_episodes)} episodes with detailed content")
    print(f"Generated dialogue for {len(dialogues)} episodes")
    print(f"All story files saved to directory: {story_dir}")
    
    # Show translation results
    if "translations" in story_data:
        print(f"Episodes translated to {list(story_data['translations'].keys())[0]}")
    elif translated_files:
        print(f"Story translated to {len(translated_files)} languages:")
        for language, file_path in translated_files:
            print(f"  - {language}: {os.path.basename(file_path)}")
    
    return story_data, story_dir

def main():
    """Main function to run the story generation pipeline"""
    parser = argparse.ArgumentParser(description="Generate a story using AI")
    parser.add_argument("topic", help="Topic or theme for the story")
    parser.add_argument("--episodes", type=int, default=5, help="Number of episodes (default: 5)")
    parser.add_argument("--type", default="general", help="Story type (e.g., mystery, sci-fi, fantasy)")
    parser.add_argument("--translate", nargs='+', help="Translate the final story to these languages (e.g., Hindi French Spanish)")
    
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
