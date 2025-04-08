import streamlit as st
import os
import json
import time
from datetime import datetime
import concurrent.futures
from typing import List, Dict, Any
import re
import zipfile
import shutil
import asyncio

# Import local components
from outline_generation_agent import OutlineGenerator
from character_development_agent import CharacterDevelopmentAgent
from plot_selector import PlotSelectorAgent, PlotResponse
from splitter_agent import StorySplitterAgent, Episode
from enhancement import EpisodeLengtheningAgent
from dialogue_generation_agent import DialogueAgent
from translator_agent import TranslatorAgent
from text_to_speech_agent import TextToSpeechAgent

# Below the existing imports

# Page configuration
st.set_page_config(
    page_title="KUKUFM Story Generator",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS for styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5em;
        font-weight: bold;
        margin-bottom: 0.5em;
    }
    .sub-header {
        font-size: 1.5em;
        font-weight: semibold;
        margin: 1em 0 0.5em 0;
    }
    .episode-title {
        font-size: 1.2em;
        font-weight: bold;
        margin-bottom: 0.5em;
    }
    .character-card {
        background-color: #8f2c61;
        border-radius: 10px;
        padding: 15px;
        margin-bottom: 10px;
    }
    .cliffhanger {
        font-style: italic;
        color: #d73b3e;
        margin-top: 10px;
    }
    .step-complete {
        color: #4CAF50;
    }
    .step-active {
        color: #2196F3;
        font-weight: bold;
    }
    .step-pending {
        color: #9E9E9E;
    }
    .container {
        padding: 20px;
        border-radius: 5px;
        margin: 10px 0;
    }
    .feedback-box {
        border: 1px solid #ddd;
        border-radius: 5px;
        padding: 15px;
        margin-top: 10px;
    }
    .download-btn {
        margin: 5px;
    }
</style>
""", unsafe_allow_html=True)

# Function to create story directory
def create_story_directory(topic):
    """Create a unique directory for this story based on name and timestamp"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # Sanitize story title for directory name
    title_slug = "".join(c if c.isalnum() else "_" for c in topic[:30])
    story_dir = f"stories/{title_slug}_{timestamp}"
    
    # Create main directory
    os.makedirs(story_dir, exist_ok=True)
    
    # Create subdirectories for different outputs
    os.makedirs(os.path.join(story_dir, "episodes"), exist_ok=True)
    os.makedirs(os.path.join(story_dir, "dialogue"), exist_ok=True)
    os.makedirs(os.path.join(story_dir, "translations"), exist_ok=True)
    
    return story_dir

# Function to save story data and final files
def save_story(story_data, story_dir):
    """Save generated story to JSON and text files in the story directory"""
    # Save complete story data as JSON using custom encoder
    with open(f"{story_dir}/story_data.json", 'w', encoding='utf-8') as f:
        json.dump(story_data, f, indent=2, ensure_ascii=False)
        
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
            # Check if episode is a dictionary or an Episode object
            if isinstance(episode, dict):
                episode_num = episode['number']
                episode_title = episode['title']
                episode_content = episode['content']
                episode_cliffhanger = episode.get('cliffhanger', '')
            else:
                episode_num = episode.number
                episode_title = episode.title
                episode_content = episode.content
                episode_cliffhanger = getattr(episode, 'cliffhanger', '')
                
            f.write(f"### Episode {episode_num}: {episode_title}\n\n")
            
            # Use enhanced content if available
            content = story_data.get('enhanced_episodes', {}).get(
                episode_num, {}).get('lengthened_content', episode_content)
            f.write(f"{content}\n\n")
            
            # Add dialogue if available
            dialogue = story_data.get('dialogue', {}).get(episode_num)
            if dialogue:
                f.write("## Dialogue\n")
                f.write(f"{dialogue}\n\n")
            
            if episode_cliffhanger:
                f.write(f"**Cliffhanger:** {episode_cliffhanger}\n\n")
    
    # Save the final story with dialogues
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
        
        # Ensure episodes are in correct order by sorting them
        sorted_episodes = sorted(story_data['episodes'], 
                                key=lambda ep: ep['number'] if isinstance(ep, dict) else ep.number)
        
        for episode in sorted_episodes:
            # Check if episode is a dictionary or an Episode object
            if isinstance(episode, dict):
                episode_num = episode['number']
                episode_title = episode['title']
            else:
                episode_num = episode.number
                episode_title = episode.title
            
            # Get dialogue for this episode
            dialogue = story_data.get('dialogue', {}).get(episode_num, "")
            
            # If no dialogue, get enhanced content
            if not dialogue:
                enhanced_episode = story_data.get('enhanced_episodes', {}).get(episode_num)
                if enhanced_episode:
                    if isinstance(enhanced_episode, dict) and 'lengthened_content' in enhanced_episode:
                        dialogue = enhanced_episode['lengthened_content']
                    elif hasattr(enhanced_episode, 'lengthened_content'):
                        dialogue = enhanced_episode.lengthened_content
                    else:
                        dialogue = episode['content'] if isinstance(episode, dict) else episode.content
                else:
                    dialogue = episode['content'] if isinstance(episode, dict) else episode.content
            
            f.write(f"### Episode {episode_num}: {episode_title}\n\n")
            f.write(f"{dialogue}\n\n")
            
            # Add a separator between episodes
            if episode_num < len(sorted_episodes):
                f.write("---\n\n")
        
        # Add metadata at the end
        f.write(f"\n\n*Generated on {datetime.now().strftime('%Y-%m-%d')}*\n")
    
    return story_dir

# Add this function with the other functions

def generate_episode_audio(episode_number):
    """Generate audio for a specific episode"""
    if not hasattr(st.session_state, 'story_dir') or not st.session_state.story_dir:
        st.error("Story directory not initialized. Cannot generate audio.")
        return None
        
    if not hasattr(st.session_state, 'episodes'):
        st.error("No episodes available. Generate episodes first.")
        return None
        
    # Find the episode
    episode = next((ep for ep in st.session_state.episodes if ep.number == episode_number), None)
    if not episode:
        st.error(f"Episode {episode_number} not found.")
        return None
    
    # Get enhanced content if available
    content = episode.content
    enhanced_episode = st.session_state.enhanced_episodes.get(episode_number)
    if enhanced_episode:
        if hasattr(enhanced_episode, 'lengthened_content'):
            content = enhanced_episode.lengthened_content
        elif isinstance(enhanced_episode, dict) and 'lengthened_content' in enhanced_episode:
            content = enhanced_episode['lengthened_content']
    
    # Initialize the TTS agent
    try:
        tts_agent = TextToSpeechAgent()
        
        with st.spinner(f"Generating audio for Episode {episode_number}..."):
            # Generate audio
            audio_path = tts_agent.generate_episode_audio(
                content, 
                episode_number, 
                st.session_state.story_dir
            )
            
        if audio_path and os.path.exists(audio_path):
            st.success(f"Audio generated for Episode {episode_number}!")
            return audio_path
        else:
            st.error("Failed to generate audio.")
            return None
            
    except Exception as e:
        st.error(f"Error generating audio: {str(e)}")
        return None

# Add this new function to play audio for an episode with streaming support
def play_episode_audio(episode_number):
    """Play or generate audio for a specific episode if not already generated"""
    # Check if required directories and episodes exist
    if not hasattr(st.session_state, 'story_dir') or not st.session_state.story_dir:
        st.error("Story directory not initialized. Cannot play audio.")
        return None
        
    if not hasattr(st.session_state, 'episodes'):
        st.error("No episodes available. Generate episodes first.")
        return None
        
    # Find the episode
    episode = next((ep for ep in st.session_state.episodes if ep.number == episode_number), None)
    if not episode:
        st.error(f"Episode {episode_number} not found.")
        return None
    
    # Check if audio file already exists
    audio_dir = os.path.join(st.session_state.story_dir, "audio")
    os.makedirs(audio_dir, exist_ok=True)  # Ensure audio directory exists
    audio_path = os.path.join(audio_dir, f"episode_{episode_number}.mp3")
    
    # Get content to speak
    content = episode.content
    enhanced_episode = st.session_state.enhanced_episodes.get(episode_number)
    if enhanced_episode:
        if hasattr(enhanced_episode, 'lengthened_content'):
            content = enhanced_episode.lengthened_content
        elif isinstance(enhanced_episode, dict) and 'lengthened_content' in enhanced_episode:
            content = enhanced_episode['lengthened_content']
    
    # Option to stream or play from file
    col1, col2 = st.columns(2)
    
    with col1:
        # Generate and play from file
        if os.path.exists(audio_path):
            st.audio(audio_path)
            st.success("Playing pre-generated audio file")
            return audio_path
        else:
            # Generate audio and play
            with st.spinner(f"Generating audio for Episode {episode_number}..."):
                audio_path = generate_episode_audio(episode_number)
            
            if audio_path and os.path.exists(audio_path):
                st.audio(audio_path)
                st.success("Audio generated and playing")
                return audio_path
    
    with col2:
        # Stream directly (uses less storage but requires active internet)
        stream_button = st.button("🔊 Stream Audio (Real-time)", key=f"stream_{episode_number}")
        
        if stream_button:
            # Create a placeholder to show streaming status
            stream_status = st.empty()
            stream_status.info("Initializing audio stream...")
            
            try:
                # Using the streaming function
                tts_agent = TextToSpeechAgent()
                
                # Create and run the async function
                async def stream_episode():
                    stream_status.info("Streaming audio... (please wait)")
                    result = await tts_agent.play_episode_audio(content)
                    if result:
                        stream_status.success("Audio streaming completed")
                    else:
                        stream_status.error("Failed to stream audio")
                
                # This creates a new event loop and runs the coroutine
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                loop.run_until_complete(stream_episode())
                loop.close()
                
                return True
                
            except Exception as e:
                st.error(f"Error streaming audio: {str(e)}")
                return None
    
    # Fallback error message if neither option works
    st.error("No audio could be generated. Please check your internet connection and API key.")
    st.info("Make sure you have the OpenAI Python package installed: `pip install openai python-dotenv`")
    return None

# Initialize session state
def init_session_state():
    # Initialize all required session state variables with default values
    if 'story_data' not in st.session_state:
        st.session_state.story_data = {}
    if 'story_dir' not in st.session_state:
        st.session_state.story_dir = None
    if 'current_step' not in st.session_state:
        st.session_state.current_step = 0
    if 'outline' not in st.session_state:
        st.session_state.outline = []
    if 'plot' not in st.session_state:
        st.session_state.plot = None
    if 'characters' not in st.session_state:
        st.session_state.characters = []
    if 'episodes' not in st.session_state:
        st.session_state.episodes = []
    if 'enhanced_episodes' not in st.session_state:
        st.session_state.enhanced_episodes = {}
    if 'dialogues' not in st.session_state:
        st.session_state.dialogues = {}
    if 'show_outline_feedback' not in st.session_state:
        st.session_state.show_outline_feedback = False
    if 'show_character_feedback' not in st.session_state:
        st.session_state.show_character_feedback = False
    if 'translated_files' not in st.session_state:
        st.session_state.translated_files = []
    if 'target_languages' not in st.session_state:
        st.session_state.target_languages = []

# Show step status indicator
def show_steps():
    steps = [
        "1. Generate Outline",
        "2. Develop Plot",
        "3. Create Characters",
        "4. Split into Episodes",
        "5. Enhance Episodes",
        "6. Generate Dialogue",
        "7. Translate Story",
        "8. Final Story"
    ]
    
    # Create columns for each step
    cols = st.columns(len(steps))
    
    for i, (col, step) in enumerate(zip(cols, steps)):
        if i < st.session_state.current_step:
            col.markdown(f"<p class='step-complete'>✓ {step}</p>", unsafe_allow_html=True)
        elif i == st.session_state.current_step:
            col.markdown(f"<p class='step-active'>➤ {step}</p>", unsafe_allow_html=True)
        else:
            col.markdown(f"<p class='step-pending'>{step}</p>", unsafe_allow_html=True)
    
    st.divider()

# Step 1: Generate Outline
def generate_outline(topic: str):
    st.session_state.current_step = 0
    
    with st.spinner("Generating story outline..."):
        outline_generator = OutlineGenerator()
        st.session_state.outline = outline_generator.generate_outline(topic)
    
    st.success("Outline generated!")
    st.session_state.current_step = 1
    return st.session_state.outline

# Handle outline feedback
def handle_outline_feedback(topic: str, feedback: str):
    with st.spinner("Refining outline based on your feedback..."):
        outline_generator = OutlineGenerator()
        
        # Clean the outline to remove any existing numbering before sending for refinement
        cleaned_outline = []
        for item in st.session_state.outline:
            # Remove any leading numbering pattern (e.g., "1. ", "2. ", etc.)
            cleaned_item = re.sub(r'^\d+\.\s*', '', item.strip())
            cleaned_outline.append(cleaned_item)
        
        refined_outline = outline_generator.refine_outline(topic, cleaned_outline, feedback)
        st.session_state.outline = refined_outline
    
    st.success("Outline refined!")

# Step 2: Develop Plot
def develop_plot():
    with st.spinner("Developing detailed plot..."):
        plot_agent = PlotSelectorAgent()
        plot_result = plot_agent.generate_plot(st.session_state.outline)
        st.session_state.plot = plot_result.detailed_plot
        st.session_state.literary_elements = plot_result.literary_elements
    
    st.success("Plot developed!")
    st.session_state.current_step = 2
    return st.session_state.plot, st.session_state.literary_elements

# Step 3: Create Characters
def create_characters():
    with st.spinner("Developing characters..."):
        character_agent = CharacterDevelopmentAgent()
        st.session_state.characters = character_agent.generate_characters(st.session_state.plot)
    
    st.success("Characters created!")
    st.session_state.current_step = 3
    return st.session_state.characters

# Handle character feedback
def handle_character_feedback(feedback: str):
    with st.spinner("Refining characters based on your feedback..."):
        character_agent = CharacterDevelopmentAgent()
        refined_characters = character_agent.refine_characters(
            st.session_state.plot, 
            st.session_state.characters, 
            feedback
        )
        st.session_state.characters = refined_characters
    
    st.success("Characters refined!")

# Step 4: Split into Episodes
def split_into_episodes(num_episodes: int):
    with st.spinner("Splitting story into episodes..."):
        splitter = StorySplitterAgent()
        st.session_state.episodes = splitter.split_story(
            st.session_state.plot, 
            st.session_state.characters, 
            num_episodes=num_episodes
        )
    
    st.success("Episodes created!")
    st.session_state.current_step = 4
    return st.session_state.episodes

# Step 5: Enhance Episodes
def enhance_episodes():
    st.write("Enhancing episodes with detailed content in parallel...")
    
    # Create progress bar
    progress_bar = st.progress(0)
    
    # Create a status area for overall progress
    status_text = st.empty()
    status_text.text("Preparing episode contexts...")
    
    lengthener = EpisodeLengtheningAgent()
    enhanced_episodes = {}
    
    # Pre-compute episode contexts
    episode_contexts = []
    previous_episodes_summary = ""
    previous_cliffhanger = ""
    
    for i, episode in enumerate(st.session_state.episodes):
        # Generate future episodes outlines to provide context
        future_episodes_outlines = ""
        if i < len(st.session_state.episodes) - 1:
            # Include up to 3 future episodes or whatever is available
            future_episodes = st.session_state.episodes[i+1:i+4]
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
            "characters": st.session_state.characters
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
            characters=context["characters"]
        )
        
        # If story_dir exists, save to file
        if hasattr(st.session_state, 'story_dir') and st.session_state.story_dir:
            file_name = f"Episode_{episode.number}_{episode.title.replace(' ', '_')}.md"
            output_path = os.path.join(st.session_state.story_dir, "episodes", file_name)
            lengthener.save_episode_to_file(enhanced, output_path)
        
        return episode.number, enhanced
    
    # Create a placeholder for episode statuses
    episode_statuses = st.empty()
    episode_status_texts = {episode.number: "Pending" for episode in st.session_state.episodes}
    
    # Process episodes in parallel
    status_text.text("Enhancing episodes in parallel...")
    
    # Determine number of workers - limit based on available processors and API rate limits
    max_workers = min(10, len(episode_contexts))  # Limiting to 3 concurrent API calls
    
    # Shared variable for completed episodes count
    completed_episodes = 0
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all enhancement tasks
        future_to_episode = {executor.submit(enhance_episode, context): context for context in episode_contexts}
        
        # Display initial status
        episode_status_md = "### Episode Enhancement Status\n"
        for episode_num, status in episode_status_texts.items():
            episode_title = next((ep.title for ep in st.session_state.episodes if ep.number == episode_num), "")
            episode_status_md += f"- Episode {episode_num} ({episode_title}): {status}\n"
        episode_statuses.markdown(episode_status_md)
        
        # Process results as they complete
        for future in concurrent.futures.as_completed(future_to_episode):
            try:
                # Get the context for this future
                context = future_to_episode[future]
                episode_num = context["episode_number"]
                
                # Get the result
                episode_number, enhanced = future.result()
                enhanced_episodes[episode_number] = enhanced
                
                # Update status
                completed_episodes += 1
                episode_status_texts[episode_num] = "✅ Completed"
                
                # Update status display
                episode_status_md = "### Episode Enhancement Status\n"
                for ep_num, status in episode_status_texts.items():
                    ep_title = next((ep.title for ep in st.session_state.episodes if ep.number == ep_num), "")
                    episode_status_md += f"- Episode {ep_num} ({ep_title}): {status}\n"
                episode_statuses.markdown(episode_status_md)
                
                # Update progress bar
                progress_bar.progress(completed_episodes / len(episode_contexts))
                
            except Exception as e:
                # Get the context for this future
                context = future_to_episode[future]
                episode_num = context["episode_number"]
                
                # Update status with error
                episode_status_texts[episode_num] = f"❌ Error: {str(e)}"
                episode_status_md = "### Episode Enhancement Status\n"
                for ep_num, status in episode_status_texts.items():
                    ep_title = next((ep.title for ep in st.session_state.episodes if ep.number == ep_num), "")
                    episode_status_md += f"- Episode {ep_num} ({ep_title}): {status}\n"
                episode_statuses.markdown(episode_status_md)
    
    st.session_state.enhanced_episodes = enhanced_episodes
    st.session_state.current_step = 5
    progress_bar.empty()
    status_text.empty()
    episode_statuses.empty()
    
    st.success(f"All {len(enhanced_episodes)} episodes enhanced successfully!")
    return st.session_state.enhanced_episodes

# Step 6: Generate Dialogue
def generate_dialogue(story_type: str):
    st.write("Generating dialogue for episodes in parallel...")
    
    # Create progress bar
    progress_bar = st.progress(0)
    
    # Create a status area for overall progress
    status_text = st.empty()
    status_text.text("Preparing to generate dialogue...")
    
    dialogue_agent = DialogueAgent()
    dialogues = {}
    
    # Create episode data for parallel processing
    episode_data = []
    for episode in st.session_state.episodes:
        # Use enhanced content if available
        enhanced_content = ""
        enhanced_episode = st.session_state.enhanced_episodes.get(episode.number)
        
        if enhanced_episode:
            if hasattr(enhanced_episode, 'lengthened_content'):
                enhanced_content = enhanced_episode.lengthened_content
            elif isinstance(enhanced_episode, dict) and 'lengthened_content' in enhanced_episode:
                enhanced_content = enhanced_episode['lengthened_content']
            else:
                enhanced_content = episode.content
        else:
            enhanced_content = episode.content
        
        # Ensure we have content to work with
        if not enhanced_content:
            enhanced_content = episode.content
            
        # Debug info
        st.session_state.debug_info = {
            'episode_number': episode.number,
            'content_type': type(enhanced_content),
            'content_length': len(enhanced_content) if enhanced_content else 0
        }
        
        episode_data.append({
            "episode": episode,
            "content": enhanced_content,
            "characters": st.session_state.characters
        })
    
    # Function to generate dialogue for a single episode
    def generate_episode_dialogue(data):
        episode = data["episode"]
        episode_content = data["content"]
        characters = data["characters"]  # Get characters from the data passed in
        
        try:
            # Check if we have valid content to work with
            if not episode_content or len(episode_content.strip()) < 10:
                st.warning(f"Episode {episode.number} content is too short or empty. Using original episode content.")
                episode_content = episode.content
            
            # Generate dialogue
            dialogue = dialogue_agent.generate_dialogue(
                story_type=story_type,
                storyline=episode_content,
                characters=characters  # Use the passed characters
            )
            
            # Validate dialogue output
            if not dialogue or len(dialogue.strip()) < 10:
                st.warning(f"Empty or too short dialogue generated for episode {episode.number}. This might indicate an API error.")
                return episode.number, f"[Dialogue generation failed for Episode {episode.number}: {episode.title}]"
            
            # Save dialogue to separate file if story_dir exists
            if hasattr(st.session_state, 'story_dir') and st.session_state.story_dir:
                dialogue_file = os.path.join(
                    st.session_state.story_dir, 
                    "dialogue", 
                    f"dialogue_episode_{episode.number}.md"
                )
                with open(dialogue_file, 'w', encoding='utf-8') as f:
                    f.write(f"# Dialogue for Episode {episode.number}: {episode.title}\n\n")
                    f.write(dialogue)
            
            return episode.number, dialogue
        except Exception as e:
            error_msg = f"Error generating dialogue for episode {episode.number}: {str(e)}"
            st.error(error_msg)
            # Also log to session state for debugging
            if 'error_log' not in st.session_state:
                st.session_state.error_log = []
            st.session_state.error_log.append({
                'episode': episode.number,
                'error': str(e),
                'content_length': len(episode_content) if episode_content else 0
            })
            return episode.number, f"[Error generating dialogue: {str(e)}]"
    
    # Create a placeholder for episode statuses
    episode_statuses = st.empty()
    episode_status_texts = {episode.number: "Pending" for episode in st.session_state.episodes}
    
    # Process dialogues in parallel
    status_text.text("Generating dialogue in parallel...")
    
    # Determine number of workers - limit based on available processors and API rate limits
    max_workers = min(len(episode_data), 10)  # Limiting to 3 concurrent API calls
    
    # Shared variable for completed episodes count
    completed_episodes = 0
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all dialogue generation tasks
        future_to_episode = {executor.submit(generate_episode_dialogue, data): data for data in episode_data}
        
        # Display initial status
        episode_status_md = "### Dialogue Generation Status\n"
        for episode_num, status in episode_status_texts.items():
            episode_title = next((ep.title for ep in st.session_state.episodes if ep.number == episode_num), "")
            episode_status_md += f"- Episode {episode_num} ({episode_title}): {status}\n"
        episode_statuses.markdown(episode_status_md)
        
        # Process results as they complete
        for future in concurrent.futures.as_completed(future_to_episode):
            try:
                # Get the data for this future
                data = future_to_episode[future]
                episode = data["episode"]
                
                # Get the result
                episode_number, dialogue = future.result()
                dialogues[episode_number] = dialogue
                
                # Update status
                completed_episodes += 1
                episode_status_texts[episode.number] = "✅ Completed"
                
                # Update status display
                episode_status_md = "### Dialogue Generation Status\n"
                for ep_num, status in episode_status_texts.items():
                    ep_title = next((ep.title for ep in st.session_state.episodes if ep.number == ep_num), "")
                    episode_status_md += f"- Episode {ep_num} ({ep_title}): {status}\n"
                episode_statuses.markdown(episode_status_md)
                
                # Update progress bar
                progress_bar.progress(completed_episodes / len(episode_data))
                
            except Exception as e:
                # Get the data for this future
                data = future_to_episode[future]
                episode = data["episode"]
                
                # Update status with error
                episode_status_texts[episode.number] = f"❌ Error: {str(e)}"
                episode_status_md = "### Dialogue Generation Status\n"
                for ep_num, status in episode_status_texts.items():
                    ep_title = next((ep.title for ep in st.session_state.episodes if ep.number == ep_num), "")
                    episode_status_md += f"- Episode {ep_num} ({ep_title}): {status}\n"
                episode_statuses.markdown(episode_status_md)
    
    st.session_state.dialogues = dialogues
    st.session_state.current_step = 6
    progress_bar.empty()
    status_text.empty()
    episode_statuses.empty()
    
    st.success(f"Dialogue generated for {len(dialogues)} episodes!")
    return st.session_state.dialogues

# Step 7: Translate Story
def translate_story(target_languages: List[str]):
    if not target_languages:
        st.warning("No languages selected for translation. Skipping.")
        st.session_state.current_step = 7  # Move to final story
        return []

    # Check if story directory exists
    if not hasattr(st.session_state, 'story_dir') or not st.session_state.story_dir:
        st.error("Story directory not initialized. Cannot translate.")
        return []
    
    st.write("Translating story to multiple languages in parallel...")
    
    # Create progress bar
    progress_bar = st.progress(0)
    
    # Create a status area for overall progress
    status_text = st.empty()
    status_text.text("Preparing to translate...")
    
    # Create path for translation directory
    translations_dir = os.path.join(st.session_state.story_dir, "translations")
    os.makedirs(translations_dir, exist_ok=True)
    
    # Read the content of the final story from the story directory
    final_story_file = os.path.join(st.session_state.story_dir, "final_story.md")
    
    # Check if the file exists
    if not os.path.exists(final_story_file):
        status_text.text("Final story file not found. Creating from available content...")
        
        # Create a combined story from available content
        try:
            final_story_content = create_combined_story_for_translation()
            
            # Save the combined content to the final story file
            with open(final_story_file, 'w', encoding='utf-8') as f:
                f.write(final_story_content)
                
            status_text.text("Created final story content for translation.")
        except Exception as e:
            st.error(f"Error creating final story content: {str(e)}")
            return []
    
    # Read the content of the final story
    with open(final_story_file, 'r', encoding='utf-8') as f:
        story_content = f.read()
        
    if not story_content or len(story_content) < 100:
        st.error("Story content is too short or empty. Cannot translate.")
        return []
    
    # Initialize translator agent
    translator = TranslatorAgent()
    translated_files = []
    
    # Function to translate to a single language
    def translate_to_language(language):
        try:
            # Translate the story
            translated_content = translator.translate_story(story_content, language)
            
            # Save the translated content to a new file
            translated_file = os.path.join(translations_dir, f"final_story_{language.lower()}.md")
            with open(translated_file, 'w', encoding='utf-8') as f:
                f.write(translated_content)
            
            # Also translate individual episodes if they exist
            if st.session_state.episodes:
                try:
                    # Create a subdirectory for translated episodes
                    translated_episodes_dir = os.path.join(st.session_state.story_dir, f"translated_{language.lower()}")
                    os.makedirs(translated_episodes_dir, exist_ok=True)
                    
                    # Translate episodes in parallel (one language at a time)
                    translate_episodes_for_language(language, translated_episodes_dir)
                except Exception as e:
                    st.warning(f"Failed to translate individual episodes to {language}: {str(e)}")
            
            return language, translated_file, None  # No error
        except Exception as e:
            return language, None, str(e)  # Return error
    
    # Function to translate individual episodes for a language
    def translate_episodes_for_language(language, translated_dir):
        episode_statuses = {}
        
        # Function to translate a single episode
        def translate_episode(episode):
            try:
                # Get enhanced content if available
                enhanced_episode = st.session_state.enhanced_episodes.get(episode.number)
                enhanced_content = ""
                
                if enhanced_episode:
                    if hasattr(enhanced_episode, 'lengthened_content'):
                        enhanced_content = enhanced_episode.lengthened_content
                    elif isinstance(enhanced_episode, dict) and 'lengthened_content' in enhanced_episode:
                        enhanced_content = enhanced_episode['lengthened_content']
                
                if not enhanced_content:
                    enhanced_content = episode.content
                
                dialogue_content = st.session_state.dialogues.get(episode.number, "")
                
                # Combine content for translation
                episode_content = (
                    f"# Episode {episode.number}: {episode.title}\n\n"
                    f"{enhanced_content}\n\n"
                    f"## Dialogue\n\n{dialogue_content}" if dialogue_content else ""
                )
                
                # Translate the content
                translated_content = translator.translate_story(episode_content, language)
                
                # Save to file
                output_file = os.path.join(
                    translated_dir, 
                    f"episode_{episode.number}_{language.lower()}.md"
                )
                with open(output_file, 'w', encoding='utf-8') as f:
                    f.write(translated_content)
                
                return episode.number, {
                    "translated_content": translated_content,
                    "file_path": output_file
                }
            except Exception as e:
                return episode.number, {"error": str(e)}
        
        # Limit concurrent API calls
        max_workers = min(10, len(st.session_state.episodes))
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all translation tasks
            future_to_episode = {
                executor.submit(translate_episode, episode): episode 
                for episode in st.session_state.episodes
            }
            
            # Process results as they complete
            for future in concurrent.futures.as_completed(future_to_episode):
                try:
                    episode_number, result = future.result()
                    episode_statuses[episode_number] = result
                except Exception as e:
                    episode = future_to_episode[future]
                    episode_statuses[episode.number] = {"error": str(e)}
        
        # Create a combined translated file with all episodes
        combined_file = os.path.join(st.session_state.story_dir, f"full_story_{language.lower()}.md")
        
        try:
            with open(combined_file, 'w', encoding='utf-8') as f:
                for episode in sorted(st.session_state.episodes, key=lambda e: e.number):
                    translation = episode_statuses.get(episode.number, {}).get('translated_content')
                    if translation:
                        f.write(f"{translation}\n\n---\n\n")
        except Exception as e:
            st.warning(f"Error creating combined translation file: {str(e)}")
        
        return episode_statuses
    
    # Create a placeholder for language statuses
    language_statuses = st.empty()
    language_status_texts = {lang: "Pending" for lang in target_languages}
    
    # Display initial status
    language_status_md = "### Translation Status\n"
    for lang, status in language_status_texts.items():
        language_status_md += f"- {lang}: {status}\n"
    language_statuses.markdown(language_status_md)
    
    # Determine number of workers - limit based on available processors and API rate limits
    max_workers = min(len(target_languages), 10)  # Limiting to 3 concurrent API calls
    
    # Shared variable for completed translations count
    completed_translations = 0
    
    # Process translations in parallel
    status_text.text("Translating in parallel...")
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all translation tasks
        future_to_language = {executor.submit(translate_to_language, lang): lang for lang in target_languages}
        
        # Process results as they complete
        for future in concurrent.futures.as_completed(future_to_language):
            try:
                language, file_path, error = future.result()
                
                # Update status
                completed_translations += 1
                
                if error:
                    language_status_texts[language] = f"❌ Error: {error}"
                    st.error(f"Error translating to {language}: {error}")
                else:
                    language_status_texts[language] = "✅ Completed"
                    translated_files.append((language, file_path))
                
                # Update status display
                language_status_md = "### Translation Status\n"
                for lang, status in language_status_texts.items():
                    language_status_md += f"- {lang}: {status}\n"
                language_statuses.markdown(language_status_md)
                
                # Update progress bar
                progress_bar.progress(completed_translations / len(target_languages))
                
            except Exception as e:
                # Get the language for this future
                language = future_to_language[future]
                
                # Update status with error
                language_status_texts[language] = f"❌ Error: {str(e)}"
                language_status_md = "### Translation Status\n"
                for lang, status in language_status_texts.items():
                    language_status_md += f"- {lang}: {status}\n"
                language_statuses.markdown(language_status_md)
    
    progress_bar.empty()
    status_text.empty()
    language_statuses.empty()
    
    if translated_files:
        st.success(f"Story translated to {len(translated_files)} languages!")
        
        # Update story data with translations and save it again
        try:
            # First read the existing story data from the JSON file
            story_data_path = os.path.join(st.session_state.story_dir, "story_data.json")
            with open(story_data_path, 'r', encoding='utf-8') as f:
                story_data = json.load(f)
            
            # Update the translations field
            story_data["translations"] = [lang for lang, _ in translated_files]
            
            # Write the updated story data back to the file
            class CustomEncoder(json.JSONEncoder):
                def default(self, obj):
                    if hasattr(obj, '__dict__'):
                        return obj.__dict__
                    return str(obj)
            
            # Save the updated JSON file
            with open(story_data_path, 'w', encoding='utf-8') as f:
                json.dump(story_data, f, indent=2, ensure_ascii=False, cls=CustomEncoder)
                
            # Also update the session state
            st.session_state.story_data = story_data
            
        except Exception as e:
            st.error(f"Error updating story data with translations: {str(e)}")
    
    st.session_state.translated_files = translated_files
    st.session_state.current_step = 7  # Move to final story
    return translated_files

# Helper function to create combined story content for translation if final_story.md doesn't exist yet
def create_combined_story_for_translation():
    """Create a combined story from episodes with dialogue for translation"""
    combined_content = f"# {st.session_state.topic}\n\n"
    
    # Characters introduction
    combined_content += "## Characters\n\n"
    for char in st.session_state.characters:
        combined_content += f"**{char['name']}** ({char['role']}): {char['description']}\n\n"
    
    # Episodes with dialogues
    combined_content += "## Story\n\n"
    
    # Ensure episodes are in correct order
    sorted_episodes = sorted(st.session_state.episodes, key=lambda ep: ep.number)
    
    for episode in sorted_episodes:
        episode_num = episode.number
        
        # Get dialogue for this episode
        dialogue = st.session_state.dialogues.get(episode_num, "")
        
        # If no dialogue, get enhanced content
        if not dialogue:
            enhanced_episode = st.session_state.enhanced_episodes.get(episode_num)
            if enhanced_episode:
                if isinstance(enhanced_episode, dict) and 'lengthened_content' in enhanced_episode:
                    dialogue = enhanced_episode['lengthened_content']
                elif hasattr(enhanced_episode, 'lengthened_content'):
                    dialogue = enhanced_episode.lengthened_content
                else:
                    dialogue = episode.content
            else:
                dialogue = episode.content
        
        combined_content += f"### Episode {episode_num}: {episode.title}\n\n"
        combined_content += f"{dialogue}\n\n"
        
        # Add a separator between episodes
        if episode_num < len(sorted_episodes):
            combined_content += "---\n\n"
    
    return combined_content

# Function to finalize story
def finalize_story(topic, story_type, target_languages=None, generate_audio=False):
    try:
        # Create a unique directory for this story
        story_dir = create_story_directory(topic)
        
        # Verify that the directories were created successfully
        if not os.path.exists(story_dir):
            st.error(f"Failed to create story directory: {story_dir}")
            return None, None
            
        # Check that the required subdirectories exist
        episodes_dir = os.path.join(story_dir, "episodes")
        dialogue_dir = os.path.join(story_dir, "dialogue")
        translations_dir = os.path.join(story_dir, "translations")
        audio_dir = os.path.join(story_dir, "audio")
        
        if not os.path.exists(episodes_dir):
            os.makedirs(episodes_dir, exist_ok=True)
        if not os.path.exists(dialogue_dir):
            os.makedirs(dialogue_dir, exist_ok=True)
        if not os.path.exists(translations_dir):
            os.makedirs(translations_dir, exist_ok=True)
        if not os.path.exists(audio_dir):
            os.makedirs(audio_dir, exist_ok=True)
            
        # Set the session state after directory verification
        st.session_state.story_dir = story_dir
        
        # Prepare serializable enhanced episodes
        serializable_enhanced_episodes = {}
        for num, ep in st.session_state.enhanced_episodes.items():
            if hasattr(ep, 'lengthened_content'):
                serializable_enhanced_episodes[num] = {
                    "lengthened_content": ep.lengthened_content,
                    "engagement_points": getattr(ep, 'engagement_points', []),
                    "summary": getattr(ep, 'summary', "")
                }
            else:
                serializable_enhanced_episodes[num] = ep
        
        # Prepare serializable episodes - converting Episode objects to dictionaries
        serializable_episodes = []
        for episode in st.session_state.episodes:
            serializable_episode = {
                "number": episode.number,
                "title": episode.title,
                "content": episode.content,
                "cliffhanger": episode.cliffhanger if hasattr(episode, 'cliffhanger') else "",
                "has_audio": False  # Default value, will be updated if audio is generated
            }
            serializable_episodes.append(serializable_episode)
        
        # Prepare serializable dialogues
        serializable_dialogues = {}
        for ep_num, dialogue in st.session_state.dialogues.items():
            serializable_dialogues[str(ep_num)] = dialogue
        
        # Create story data structure
        story_data = {
            "topic": topic,
            "story_type": story_type,
            "outline": st.session_state.outline,
            "detailed_plot": st.session_state.plot,
            "literary_elements": st.session_state.literary_elements if hasattr(st.session_state, 'literary_elements') else {},
            "characters": st.session_state.characters,
            "episodes": serializable_episodes,  # Use serializable episodes instead of Episode objects
            "enhanced_episodes": serializable_enhanced_episodes,
            "dialogue": serializable_dialogues,
            "generated_at": datetime.now().isoformat(),
            "translations": [lang for lang in target_languages] if target_languages else [],
            "audio_generated": generate_audio
        }
        
        # Generate audio for episodes if requested
        if generate_audio:
            try:
                st.write("Generating audio for episodes...")
                tts_agent = TextToSpeechAgent()
                
                for i, episode in enumerate(st.session_state.episodes):
                    with st.spinner(f"Generating audio for Episode {episode.number}..."):
                        # Get enhanced content if available
                        content = episode.content
                        enhanced_episode = serializable_enhanced_episodes.get(episode.number)
                        if enhanced_episode and 'lengthened_content' in enhanced_episode:
                            content = enhanced_episode['lengthened_content']
                        
                        # Generate audio
                        audio_path = tts_agent.generate_episode_audio(
                            content, 
                            episode.number, 
                            story_dir
                        )
                        
                        # Update the serializable episode with audio info
                        if audio_path and os.path.exists(audio_path):
                            serializable_episodes[i]["has_audio"] = True
                
                # Update story data with audio info
                story_data["episodes"] = serializable_episodes
                
            except Exception as e:
                st.warning(f"Audio generation encountered an issue: {str(e)}")
        
        # Ensure the JSON file is saved correctly using a custom encoder for non-serializable objects
        class CustomEncoder(json.JSONEncoder):
            def default(self, obj):
                if hasattr(obj, '__dict__'):
                    return obj.__dict__
                return str(obj)
        
        # Save the JSON file with the custom encoder
        story_data_path = os.path.join(story_dir, "story_data.json")
        with open(story_data_path, 'w', encoding='utf-8') as f:
            json.dump(story_data, f, indent=2, ensure_ascii=False, cls=CustomEncoder)
        
        # Also save the story in other formats
        st.session_state.story_data = story_data
        save_story(story_data, story_dir)
        
        return story_data, story_dir
    except Exception as e:
        st.error(f"Error finalizing story: {str(e)}")
        import traceback
        st.error(traceback.format_exc())
        return None, None# Main app function
        import traceback
        st.error(traceback.format_exc())
        return None, None

# Main app function
def main():
    # Initialize session state
    init_session_state()
    
    # Display header
    st.markdown("<h1 class='main-header'>📚 KUKUFM Story Generator</h1>", unsafe_allow_html=True)
    
    st.markdown(
        """
        Create engaging multi-episode stories with rich characters and dialogue using AI.
        Complete each step of the process to generate your story.
        """
    )
    
    # Display steps indicator
    show_steps()
    
    # Story Configuration Section
    if st.session_state.current_step == 0:
        with st.form("story_config_form"):
            st.markdown("<h2 class='sub-header'>Story Configuration</h2>", unsafe_allow_html=True)
            
            topic = st.text_input("Story Topic/Theme", 
                                value="A mysterious hotel with supernatural occurrences", 
                                help="Enter a brief description of your story idea")
            
            col1, col2 = st.columns(2)
            
            with col1:
                num_episodes = st.slider("Number of Episodes", 
                                        min_value=2, 
                                        max_value=10, 
                                        value=5, 
                                        help="Select the number of episodes for your story")
                
            with col2:
                story_type = st.selectbox("Story Type", 
                                        ["novel", "drama"],
                                        index=0,
                                        help="Select the genre of your story")
            
            # Translation options
            languages = st.multiselect("Translate to languages (optional)", 
                                    ["Hindi", "Tamil", "Telugu", "Malayalam", "Kannada", "Bengali", "Marathi", "Gujarati", "Punjabi", "Urdu", "Assamese", "Odia"],
                                    help="Select Indian languages to translate your story into")
            
            # Submit button
            submit = st.form_submit_button("Generate Story")
            
            if submit:
                if not topic:
                    st.error("Please enter a story topic")
                else:
                    # Start the story generation process
                    st.session_state.topic = topic
                    st.session_state.num_episodes = num_episodes
                    st.session_state.story_type = story_type
                    st.session_state.target_languages = languages
                    
                    # Generate outline
                    outline = generate_outline(topic)
                    st.rerun()
    
    # Step 1: Outline Feedback
    elif st.session_state.current_step == 1:
        st.markdown("<h2 class='sub-header'>Story Outline</h2>", unsafe_allow_html=True)
        
        # Display outline
        for i, event in enumerate(st.session_state.outline, 1):
            st.write(f"{i}. {event}")
        
        # Feedback options
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("Accept Outline"):
                # Move to next step - Plot Development
                detailed_plot, literary_elements = develop_plot()
                st.rerun()
        
        with col2:
            if st.button("Refine Outline"):
                st.session_state.show_outline_feedback = True
        
        if st.session_state.show_outline_feedback:
            with st.form("outline_feedback_form"):
                st.write("Please provide feedback on how to improve the outline:")
                feedback = st.text_area("Feedback", height=100,
                                      help="Suggest changes, additional elements, or different directions for the story")
                
                submit_feedback = st.form_submit_button("Submit Feedback")
                
                if submit_feedback and feedback:
                    handle_outline_feedback(st.session_state.topic, feedback)
                    st.session_state.show_outline_feedback = False
                    st.rerun()
    
    # Step 2: Plot Development (no feedback needed, automatic progression)
    elif st.session_state.current_step == 2:
        st.markdown("<h2 class='sub-header'>Detailed Plot</h2>", unsafe_allow_html=True)
        
        # Show the detailed plot
        st.write(st.session_state.plot)
        
        # Show literary elements
        if hasattr(st.session_state, 'literary_elements') and st.session_state.literary_elements:
            st.markdown("<h3>Literary Elements Used</h3>", unsafe_allow_html=True)
            for category, element in st.session_state.literary_elements.items():
                st.write(f"**{category.capitalize()}:** {element}")
        
        # Progress button
        if st.button("Continue to Character Creation"):
            # Create characters and move to next step
            characters = create_characters()
            st.rerun()
    
    # Step 3: Character Development and Feedback
    elif st.session_state.current_step == 3:
        st.markdown("<h2 class='sub-header'>Characters</h2>", unsafe_allow_html=True)
        
        # Display characters in a grid
        cols = st.columns(2)  # 2 columns for character display
        
        for i, character in enumerate(st.session_state.characters):
            with cols[i % 2]:  # Alternate between columns
                with st.container():
                    st.markdown(
                        f"""
                        <div class="character-card">
                            <h3>{character['name']}</h3>
                            <p><strong>Role:</strong> {character['role']}</p>
                            <p>{character['description']}</p>
                        </div>
                        """, 
                        unsafe_allow_html=True
                    )
        
        # Feedback options
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("Accept Characters"):
                # Move to next step - Episode Splitting
                episodes = split_into_episodes(st.session_state.num_episodes)
                st.rerun()
        
        with col2:
            if st.button("Refine Characters"):
                st.session_state.show_character_feedback = True
        
        if st.session_state.show_character_feedback:
            with st.form("character_feedback_form"):
                st.write("Please provide feedback on how to improve the characters:")
                feedback = st.text_area("Feedback", height=100,
                                      help="Suggest changes to characters, new personalities, or different roles")
                
                submit_feedback = st.form_submit_button("Submit Feedback")
                
                if submit_feedback and feedback:
                    handle_character_feedback(feedback)
                    st.session_state.show_character_feedback = False
                    st.rerun()
    
    # Step 4: Episode Splitting (automatic progression)
    elif st.session_state.current_step == 4:
        st.markdown("<h2 class='sub-header'>Episode Outlines</h2>", unsafe_allow_html=True)
        
        # Show episode outlines
        for episode in st.session_state.episodes:
            with st.expander(f"Episode {episode.number}: {episode.title}"):
                st.write(episode.content)
                if episode.cliffhanger:
                    st.markdown(f"<p class='cliffhanger'>Cliffhanger: {episode.cliffhanger}</p>", unsafe_allow_html=True)

                # Add audio button for each episode
                if st.button(f"🔊 Listen to Episode {episode.number}", key=f"audio_ep4_{episode.number}"):
                    play_episode_audio(episode.number)
        
        # Progress button
        if st.button("Enhance Episodes"):
            # Enhance episodes and move to next step
            enhanced_episodes = enhance_episodes()
            st.rerun()
    
    # Step 5: Episode Enhancement (automatic progression)
    elif st.session_state.current_step == 5:
        st.markdown("<h2 class='sub-header'>Enhanced Episodes</h2>", unsafe_allow_html=True)
        
        # Sort episode numbers to ensure they display in order
        sorted_episode_nums = sorted(st.session_state.enhanced_episodes.keys())
        
        # Show enhanced episodes in correct order
        for episode_num in sorted_episode_nums:
            enhanced = st.session_state.enhanced_episodes.get(episode_num)
            
            # Find the original episode
            original_episode = next(
                (ep for ep in st.session_state.episodes if ep.number == episode_num), 
                None
            )
            
            if original_episode and enhanced:
                title = original_episode.title
                with st.expander(f"Episode {episode_num}: {title}"):
                    if hasattr(enhanced, 'lengthened_content'):
                        st.write(enhanced.lengthened_content)
                    else:
                        st.write(enhanced.get('lengthened_content', 'Content not available'))
                    if st.button(f"🔊 Listen to Episode {episode_num}", key=f"audio_ep5_{episode_num}"):
                        play_episode_audio(episode_num)
        
        # Progress button
        if st.button("Generate Dialogue"):
            # Generate dialogue and move to next step
            dialogues = generate_dialogue(st.session_state.story_type)
            st.rerun()
    
    # Step 6: Dialogue Generation
    elif st.session_state.current_step == 6:
        st.markdown("<h2 class='sub-header'>Story with Dialogue</h2>", unsafe_allow_html=True)
        
        # Debug info for troubleshooting
        if hasattr(st.session_state, 'debug_info'):
            with st.expander("Debug Information"):
                st.json(st.session_state.debug_info)
        
        # Show error logs if any
        if hasattr(st.session_state, 'error_log') and st.session_state.error_log:
            with st.expander("Error Logs"):
                st.error("Errors occurred during dialogue generation:")
                for error in st.session_state.error_log:
                    st.write(f"Episode {error['episode']}: {error['error']}")
                    st.write(f"Content length: {error['content_length']}")
                    st.divider()
        
        # Show tabs for episodes with dialogue
        if st.session_state.episodes:
            # Sort episodes by episode number to ensure proper ordering
            sorted_episodes = sorted(st.session_state.episodes, key=lambda ep: ep.number)
            tab_titles = [f"Episode {ep.number}: {ep.title}" for ep in sorted_episodes]
            tabs = st.tabs(tab_titles)
            
            for i, tab in enumerate(tabs):
                with tab:
                    episode = sorted_episodes[i]
                    episode_num = episode.number
                    
                    # Display episode content
                    enhanced = st.session_state.enhanced_episodes.get(episode_num)
                    content = ""
                    if enhanced:
                        if hasattr(enhanced, 'lengthened_content'):
                            content = enhanced.lengthened_content
                        else:
                            content = enhanced.get('lengthened_content', '')
                    
                    # # Add audio player at the top of each tab
                    # col1, col2 = st.columns([3, 1])
                    # with col1:
                    #     st.markdown("<h3>Episode Content</h3>", unsafe_allow_html=True)
                    # with col2:
                    #     if st.button(f"🔊 Listen", key=f"audio_ep6_{episode_num}"):
                    #         play_episode_audio(episode_num)
                    
                    # st.write(content)
                    
                    # Display dialogue
                    dialogue = st.session_state.dialogues.get(episode_num, "")
                    if dialogue:
                        st.markdown("<h3>Dialogue</h3>", unsafe_allow_html=True)
                        st.write(dialogue)
                    else:
                        st.warning(f"No dialogue generated for Episode {episode_num}. Dialogue generation may have failed.")
        
        # Proceed to translation button (simple transition like other steps)
        if st.button("Proceed to Translation"):
            # Simply move to the next step based on whether we have target languages
            if st.session_state.target_languages:
                st.session_state.current_step = 7
            else:
                # Skip translation if no languages selected
                st.session_state.current_step = 8
            st.rerun()

    # Step 7: Translation
    elif st.session_state.current_step == 7:
        st.markdown("<h2 class='sub-header'>Story Translation</h2>", unsafe_allow_html=True)
        
        # First create the story directory and finalize the story without translations
        if not hasattr(st.session_state, 'story_dir') or not st.session_state.story_dir:
            with st.spinner("Creating story files..."):
                story_data, story_dir = finalize_story(
                    st.session_state.topic,
                    st.session_state.story_type
                )
                if story_dir:
                    st.session_state.story_dir = story_dir
                else:
                    st.error("Failed to create story directory")
                    st.stop()
        
        # Then translate the story
        with st.spinner("Translating story..."):
            translated_files = translate_story(st.session_state.target_languages)
        
        # Move to final story step
        st.session_state.current_step = 8
        st.rerun()

    # Step 8: Downloads & Final Story
    elif st.session_state.current_step == 8:
        st.markdown("<h2 class='sub-header'>Final Story</h2>", unsafe_allow_html=True)
        
        st.success(f"Your story '{st.session_state.topic}' has been successfully generated!")
        
        # Add a section for episode audio players
        st.markdown("<h3>Listen to Episodes</h3>", unsafe_allow_html=True)
        
        # Create episode audio player buttons
        cols = st.columns(min(3, len(st.session_state.episodes)))  # Up to 3 columns
        
        for i, episode in enumerate(sorted(st.session_state.episodes, key=lambda ep: ep.number)):
            with cols[i % 3]:
                st.markdown(f"**Episode {episode.number}: {episode.title}**")
                if st.button(f"🔊 Play", key=f"audio_ep7_{episode.number}"):
                    play_episode_audio(episode.number)

        # Show download options
        st.markdown("<h3>Download Options</h3>", unsafe_allow_html=True)
        
        if st.session_state.story_dir:
            # Create a ZIP file containing all story files
            zip_path = os.path.join(st.session_state.story_dir, "story_files.zip")
            
            with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                # Add all files from the story directory
                for root, dirs, files in os.walk(st.session_state.story_dir):
                    for file in files:
                        if file != "story_files.zip":  # Don't include the zip file itself
                            file_path = os.path.join(root, file)
                            arcname = os.path.relpath(file_path, st.session_state.story_dir)
                            zipf.write(file_path, arcname)
            
            # Complete story
            final_story_path = os.path.join(st.session_state.story_dir, "final_story.md")
            
            if os.path.exists(final_story_path):
                with open(final_story_path, 'r', encoding='utf-8') as f:
                    final_story_content = f.read()
                
                st.download_button(
                    label="Download Complete Story (MD)",
                    data=final_story_content,
                    file_name="final_story.md",
                    mime="text/markdown",
                    key="download_final_story"
                )
            
            # Download ZIP file
            if os.path.exists(zip_path):
                with open(zip_path, 'rb') as f:
                    zip_content = f.read()
                
                st.download_button(
                    label="Download All Story Files (ZIP)",
                    data=zip_content,
                    file_name="story_files.zip",
                    mime="application/zip",
                    key="download_story_zip"
                )
            
            # Show translated versions if available
            if hasattr(st.session_state, 'translated_files') and st.session_state.translated_files:
                st.markdown("<h3>Translated Versions</h3>", unsafe_allow_html=True)
                
                for lang, file_path in st.session_state.translated_files:
                    if os.path.exists(file_path):
                        with open(file_path, 'r', encoding='utf-8') as f:
                            translated_content = f.read()
                        
                        st.download_button(
                            label=f"Download {lang} Version (MD)",
                            data=translated_content,
                            file_name=f"final_story_{lang.lower()}.md",
                            mime="text/markdown",
                            key=f"download_{lang.lower()}"
                        )
            
            # Show story preview
            st.markdown("<h3>Story Preview</h3>", unsafe_allow_html=True)
            
            if os.path.exists(final_story_path):
                with open(final_story_path, 'r', encoding='utf-8') as f:
                    preview_content = f.read()
                
                # Show first 1000 characters as preview
                preview = preview_content[:1000] + "..." if len(preview_content) > 1000 else preview_content
                st.text_area("Preview", preview, height=200)
        
        # Generate new story button
        if st.button("Generate New Story"):
            # Reset session state
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            init_session_state()
            st.rerun()

if __name__ == "__main__":
    main()
