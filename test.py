import os
import sys
from dotenv import load_dotenv
import json
import time

# Load environment variables
load_dotenv()

# Add checks to ensure all required modules are installed
required_modules = [
    "langchain", "langchain_core", "langchain_groq", 
    "langchain_openai", "pydantic", "openai", "yaml"
]

missing_modules = []
for module in required_modules:
    try:
        __import__(module)
    except ImportError:
        missing_modules.append(module)

if missing_modules:
    print("ERROR: The following required modules are missing:")
    for module in missing_modules:
        print(f"  - {module}")
    print("\nPlease install them using: pip install -r requirements.txt")
    sys.exit(1)

# Import local modules (with error handling)
try:
    from outline_generation_agent import OutlineGenerator
    from character_development_agent import CharacterDevelopmentAgent
    from plot_selector import StoryElementLibrary
    from splitter_agent import StorySplitterAgent
    from enhancement import EpisodeLengtheningAgent
    from dialogue_generation_agent import DialogueAgent
    from llm_api import llm_api
except ImportError as e:
    print(f"ERROR: Failed to import local modules: {e}")
    print("Make sure all required files are in the current directory.")
    sys.exit(1)

def check_api_keys():
    """Check if API keys are properly configured"""
    groq_key = os.getenv('GROQ_API_KEY')
    openai_key = os.getenv('OPENAI_API_KEY')
    
    if not groq_key:
        print("WARNING: GROQ_API_KEY is not set in .env file")
    else:
        print("✓ GROQ_API_KEY found")
        
    if not openai_key:
        print("WARNING: OPENAI_API_KEY is not set in .env file")
    else:
        print("✓ OPENAI_API_KEY found")
    
    return bool(groq_key or openai_key)

def test_llm_api():
    """Test the LLM API connection"""
    print("\n=== Testing LLM API Connection ===")
    try:
        model = llm_api(model_type="default")
        if model:
            print("✓ LLM API connection successful")
            return True
        else:
            print("✗ Failed to initialize LLM API")
            return False
    except Exception as e:
        print(f"✗ Error testing LLM API: {e}")
        return False

def test_outline_generator(topic="A mystery in an abandoned amusement park"):
    """Test the outline generator"""
    print(f"\n=== Testing Outline Generator with topic: '{topic}' ===")
    try:
        generator = OutlineGenerator()
        start_time = time.time()
        events = generator.generate_outline(topic)
        elapsed_time = time.time() - start_time
        
        if events and len(events) > 0:
            print(f"✓ Outline generation successful (took {elapsed_time:.2f} seconds)")
            print(f"✓ Generated {len(events)} events")
            return events
        else:
            print("✗ Failed to generate outline events")
            return None
    except Exception as e:
        print(f"✗ Error testing outline generator: {e}")
        return None

def test_character_development(plot):
    """Test the character development agent"""
    print(f"\n=== Testing Character Development Agent ===")
    try:
        agent = CharacterDevelopmentAgent()
        start_time = time.time()
        characters = agent.generate_characters(plot)
        elapsed_time = time.time() - start_time
        
        if characters and len(characters) > 0:
            print(f"✓ Character development successful (took {elapsed_time:.2f} seconds)")
            print(f"✓ Generated {len(characters)} characters")
            return characters
        else:
            print("✗ Failed to generate characters")
            return None
    except Exception as e:
        print(f"✗ Error testing character development agent: {e}")
        return None

def test_plot_selector(outline, story_type="mystery"):
    """Test the plot selector"""
    print(f"\n=== Testing Plot Selector with story type: '{story_type}' ===")
    try:
        library = StoryElementLibrary()
        start_time = time.time()
        plot_options = library.generate_plot_options(outline, story_type)
        elapsed_time = time.time() - start_time
        
        if plot_options and len(plot_options) > 0:
            print(f"✓ Plot selection successful (took {elapsed_time:.2f} seconds)")
            print(f"✓ Generated {len(plot_options)} plot options")
            
            # Select 3 random plots for further testing
            import random
            selected_plots = random.sample(plot_options, min(3, len(plot_options)))
            print("\nSelected plots for testing:")
            for i, plot in enumerate(selected_plots, 1):
                print(f"{i}. {plot}")
                
            return selected_plots
        else:
            print("✗ Failed to generate plot options")
            return None
    except Exception as e:
        print(f"✗ Error testing plot selector: {e}")
        return None

def test_story_splitter(outline, plots, characters):
    """Test the story splitter agent"""
    print(f"\n=== Testing Story Splitter Agent ===")
    try:
        splitter = StorySplitterAgent()
        
        # Create a combined outline from individual events
        if isinstance(outline, list):
            combined_outline = " ".join(outline)
        else:
            combined_outline = outline
            
        start_time = time.time()
        episodes = splitter.split_story(combined_outline, plots, characters, num_episodes=3)
        elapsed_time = time.time() - start_time
        
        if episodes and len(episodes) > 0:
            print(f"✓ Story splitting successful (took {elapsed_time:.2f} seconds)")
            print(f"✓ Generated {len(episodes)} episodes")
            return episodes
        else:
            print("✗ Failed to split story into episodes")
            return None
    except Exception as e:
        print(f"✗ Error testing story splitter agent: {e}")
        return None

def test_episode_enhancer(episode, characters):
    """Test the episode enhancer agent"""
    print(f"\n=== Testing Episode Enhancer Agent ===")
    try:
        enhancer = EpisodeLengtheningAgent()
        
        if not episode:
            print("✗ No episode provided for enhancement")
            return None
            
        # Create a sample episode for enhancement
        episode_title = episode.title
        episode_number = episode.number
        episode_outline = episode.content
        
        start_time = time.time()
        # For testing, we'll set shorter content to speed up the test
        enhanced = enhancer.lengthen_episode(
            episode_title=episode_title,
            episode_number=episode_number,
            episode_outline=episode_outline,
            previous_episodes_summary="This is the first episode.",
            previous_cliffhanger="",
            include_cliffhanger=bool(episode.cliffhanger)
        )
        elapsed_time = time.time() - start_time
        
        if enhanced and enhanced.lengthened_content:
            print(f"✓ Episode enhancement successful (took {elapsed_time:.2f} seconds)")
            word_count = len(enhanced.lengthened_content.split())
            print(f"✓ Generated enhanced content with {word_count} words")
            return enhanced
        else:
            print("✗ Failed to enhance episode")
            return None
    except Exception as e:
        print(f"✗ Error testing episode enhancer: {e}")
        return None

def test_dialogue_generator(episode, characters):
    """Test the dialogue generator agent"""
    print(f"\n=== Testing Dialogue Generator Agent ===")
    try:
        if not episode or not characters:
            print("✗ No episode or characters provided for dialogue generation")
            return None
            
        dialogue_agent = DialogueAgent()
        
        start_time = time.time()
        dialogue = dialogue_agent.generate_dialogue(
            story_type="drama",  # Use drama for testing
            storyline=episode.content,
            characters=characters
        )
        elapsed_time = time.time() - start_time
        
        if dialogue:
            print(f"✓ Dialogue generation successful (took {elapsed_time:.2f} seconds)")
            # Count dialogue lines as a rough measure of success
            line_count = dialogue.count('\n')
            print(f"✓ Generated dialogue with approximately {line_count} lines")
            return dialogue
        else:
            print("✗ Failed to generate dialogue")
            return None
    except Exception as e:
        print(f"✗ Error testing dialogue generator: {e}")
        return None

def save_test_results(results):
    """Save test results to a file"""
    try:
        filename = f"test_results_{time.strftime('%Y%m%d_%H%M%S')}.json"
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\n✓ Test results saved to {filename}")
    except Exception as e:
        print(f"\n✗ Failed to save test results: {e}")

def main():
    print("=== KUKUFM Story Generation System Test ===")
    print("This test will check all components of the story generation pipeline.")
    
    results = {
        "api_keys": check_api_keys(),
        "llm_api": None,
        "outline_generator": None,
        "character_development": None,
        "plot_selector": None,
        "story_splitter": None,
        "episode_enhancer": None,
        "dialogue_generator": None,
    }
    
    # Test LLM API
    results["llm_api"] = test_llm_api()
    
    if not results["llm_api"]:
        print("\nCannot proceed with further tests due to LLM API issues.")
        save_test_results(results)
        return
    
    # Test outline generator
    test_topic = "A mystery in an ancient library"
    outline = test_outline_generator(test_topic)
    results["outline_generator"] = bool(outline)
    
    if not results["outline_generator"]:
        print("\nCannot proceed with further tests due to outline generation issues.")
        save_test_results(results)
        return
    
    # Test character development
    outline_text = " ".join(outline) if isinstance(outline, list) else outline
    characters = test_character_development(outline_text)
    results["character_development"] = bool(characters)
    
    if not results["character_development"]:
        print("\nCannot proceed with further tests due to character development issues.")
        save_test_results(results)
        return
    
    # Test plot selector
    plots = test_plot_selector(outline_text)
    results["plot_selector"] = bool(plots)
    
    if not results["plot_selector"]:
        print("\nCannot proceed with further tests due to plot selection issues.")
        save_test_results(results)
        return
    
    # Test story splitter
    episodes = test_story_splitter(outline_text, plots, characters)
    results["story_splitter"] = bool(episodes)
    
    if not results["story_splitter"] or not episodes:
        print("\nCannot proceed with further tests due to story splitting issues.")
        save_test_results(results)
        return
    
    # Test episode enhancer (using the first episode)
    first_episode = episodes[0] if episodes else None
    enhanced_episode = test_episode_enhancer(first_episode, characters)
    results["episode_enhancer"] = bool(enhanced_episode)
    
    # Test dialogue generator (using the first episode)
    dialogue = test_dialogue_generator(first_episode, characters)
    results["dialogue_generator"] = bool(dialogue)
    
    # Print summary
    print("\n=== Test Summary ===")
    for component, status in results.items():
        print(f"{component}: {'✓ Success' if status else '✗ Failed'}")
    
    # Save results
    test_results = {
        "test_results": results,
        "topic": test_topic,
        "outline": outline,
        "characters": characters,
        "plots": plots,
        "episodes": [
            {
                "number": ep.number,
                "title": ep.title,
                "content_length": len(ep.content),
                "has_cliffhanger": bool(ep.cliffhanger.strip())
            }
            for ep in episodes
        ] if episodes else None
    }
    
    # Add enhanced episode if available
    if enhanced_episode:
        test_results["enhanced_episode"] = {
            "title": enhanced_episode.title,
            "content_length": len(enhanced_episode.lengthened_content),
            "word_count": len(enhanced_episode.lengthened_content.split())
        }
    
    # Add dialogue if available
    if dialogue:
        test_results["dialogue"] = {
            "content_length": len(dialogue),
            "line_count": dialogue.count('\n')
        }
    
    save_test_results(test_results)

if __name__ == "__main__":
    main()
