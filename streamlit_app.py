import streamlit as st
import os
import json
import time
from datetime import datetime
import concurrent.futures
from typing import List, Dict, Any

# Import local components
from outline_generation_agent import OutlineGenerator
from character_development_agent import CharacterDevelopmentAgent
from plot_selector import PlotSelectorAgent, PlotResponse
from splitter_agent import StorySplitterAgent, Episode
from enhancement import EpisodeLengtheningAgent
from dialogue_generation_agent import DialogueAgent
from translator_agent import TranslatorAgent

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
        background-color: #f0f2f6;
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
    
    # Create directories
    if not os.path.exists(story_dir):
        os.makedirs(story_dir)
    
    # Create subdirectories for different outputs
    os.makedirs(f"{story_dir}/episodes", exist_ok=True)
    os.makedirs(f"{story_dir}/dialogue", exist_ok=True)
    
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
    
    return story_dir

# Initialize session state
def init_session_state():
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

# Show step status indicator
def show_steps():
    steps = [
        "1. Generate Outline",
        "2. Develop Plot",
        "3. Create Characters",
        "4. Split into Episodes",
        "5. Enhance Episodes",
        "6. Generate Dialogue",
        "7. Final Story"
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
        refined_outline = outline_generator.refine_outline(topic, st.session_state.outline, feedback)
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
    st.write("Enhancing episodes with detailed content...")
    
    # Create progress bar
    progress_bar = st.progress(0)
    
    # Create a status area to show which episode is being processed
    status_text = st.empty()
    
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
        if st.session_state.story_dir:
            file_name = f"Episode_{episode.number}_{episode.title.replace(' ', '_')}.md"
            output_path = os.path.join(st.session_state.story_dir, "episodes", file_name)
            lengthener.save_episode_to_file(enhanced, output_path)
        
        return episode.number, enhanced
    
    # Process episodes sequentially for visibility in UI
    total_episodes = len(episode_contexts)
    for i, context in enumerate(episode_contexts):
        status_text.text(f"Enhancing Episode {context['episode_number']}: {context['episode_title']}")
        episode_number, enhanced = enhance_episode(context)
        enhanced_episodes[episode_number] = enhanced
        
        # Update progress
        progress_bar.progress((i + 1) / total_episodes)
    
    st.session_state.enhanced_episodes = enhanced_episodes
    st.session_state.current_step = 5
    progress_bar.empty()
    status_text.empty()
    
    st.success("Episodes enhanced!")
    return st.session_state.enhanced_episodes

# Step 6: Generate Dialogue
def generate_dialogue(story_type: str):
    st.write("Generating dialogue for episodes...")
    
    # Create progress bar
    progress_bar = st.progress(0)
    
    # Create a status area to show which episode is being processed
    status_text = st.empty()
    
    dialogue_agent = DialogueAgent()
    dialogues = {}
    
    # Process dialogues sequentially for visibility in UI
    total_episodes = len(st.session_state.episodes)
    for i, episode in enumerate(st.session_state.episodes):
        status_text.text(f"Generating dialogue for Episode {episode.number}: {episode.title}")
        
        # Use enhanced content if available
        enhanced_episode = st.session_state.enhanced_episodes.get(episode.number)
        if enhanced_episode and hasattr(enhanced_episode, 'lengthened_content'):
            episode_content = enhanced_episode.lengthened_content
        else:
            episode_content = episode.content
        
        # Generate dialogue
        dialogue = dialogue_agent.generate_dialogue(
            story_type=story_type,
            storyline=episode_content,
            characters=st.session_state.characters
        )
        
        # Save dialogue to separate file if story_dir exists
        if st.session_state.story_dir:
            dialogue_file = os.path.join(
                st.session_state.story_dir, 
                "dialogue", 
                f"dialogue_episode_{episode.number}.md"
            )
            with open(dialogue_file, 'w', encoding='utf-8') as f:
                f.write(f"# Dialogue for Episode {episode.number}: {episode.title}\n\n")
                f.write(dialogue)
        
        dialogues[episode.number] = dialogue
        
        # Update progress
        progress_bar.progress((i + 1) / total_episodes)
    
    st.session_state.dialogues = dialogues
    st.session_state.current_step = 6
    progress_bar.empty()
    status_text.empty()
    
    st.success("Dialogue generated!")
    return st.session_state.dialogues

# Step 7: Translate Story
def translate_story(target_languages: List[str]):
    st.write("Translating story...")
    
    # Check if any languages are selected
    if not target_languages:
        st.warning("No languages selected for translation. Skipping.")
        return []
    
    # Create progress bar
    progress_bar = st.progress(0)
    
    # Create a status area to show which translation is being processed
    status_text = st.empty()
    
    # Read the content of the final story from the story directory
    final_story_file = os.path.join(st.session_state.story_dir, "final_story.md")
    with open(final_story_file, 'r', encoding='utf-8') as f:
        story_content = f.read()
    
    # Initialize translator agent
    translator = TranslatorAgent()
    translated_files = []
    
    # Process translations sequentially for UI feedback
    total_languages = len(target_languages)
    for i, language in enumerate(target_languages):
        status_text.text(f"Translating to {language}...")
        
        try:
            # Translate the story
            translated_content = translator.translate_story(story_content, language)
            
            # Save the translated content to a new file
            translated_file = os.path.join(st.session_state.story_dir, f"final_story_{language.lower()}.md")
            with open(translated_file, 'w', encoding='utf-8') as f:
                f.write(translated_content)
            
            translated_files.append((language, translated_file))
            
        except Exception as e:
            st.error(f"Error translating to {language}: {str(e)}")
        
        # Update progress
        progress_bar.progress((i + 1) / total_languages)
    
    progress_bar.empty()
    status_text.empty()
    
    if translated_files:
        st.success(f"Story translated to {len(translated_files)} languages!")
    
    st.session_state.translated_files = translated_files
    return translated_files

# Function to finalize story
def finalize_story(topic, story_type, target_languages):
    # Create a unique directory for this story
    story_dir = create_story_directory(topic)
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
    
    # Create story data structure
    story_data = {
        "topic": topic,
        "story_type": story_type,
        "outline": st.session_state.outline,
        "detailed_plot": st.session_state.plot,
        "literary_elements": st.session_state.literary_elements if hasattr(st.session_state, 'literary_elements') else {},
        "characters": st.session_state.characters,
        "episodes": st.session_state.episodes,
        "enhanced_episodes": serializable_enhanced_episodes,
        "dialogue": st.session_state.dialogues,
        "generated_at": datetime.now().isoformat(),
        "translations": [lang for lang in target_languages] if target_languages else []
    }
    
    st.session_state.story_data = story_data
    
    # Save all story data to the story directory
    save_story(story_data, story_dir)
    
    # Translate if languages are specified
    if target_languages:
        translate_story(target_languages)
    
    return story_data, story_dir

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
                                        ["general", "mystery", "sci-fi", "fantasy", "romance", "thriller", "comedy"],
                                        index=1,
                                        help="Select the genre of your story")
            
            # Translation options
            languages = st.multiselect("Translate to languages (optional)", 
                                    ["Hindi", "Spanish", "French", "German", "Chinese", "Japanese", "Russian"],
                                    help="Select languages to translate your story into")
            
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
                    st.session_state.languages = languages
                    
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
        
        # Progress button
        if st.button("Enhance Episodes"):
            # Enhance episodes and move to next step
            enhanced_episodes = enhance_episodes()
            st.rerun()
    
    # Step 5: Episode Enhancement (automatic progression)
    elif st.session_state.current_step == 5:
        st.markdown("<h2 class='sub-header'>Enhanced Episodes</h2>", unsafe_allow_html=True)
        
        # Show enhanced episodes
        for episode_num, enhanced in st.session_state.enhanced_episodes.items():
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
        
        # Progress button
        if st.button("Generate Dialogue"):
            # Generate dialogue and move to next step
            dialogues = generate_dialogue(st.session_state.story_type)
            st.rerun()
    
    # Step 6: Dialogue Generation and Final Story
    elif st.session_state.current_step == 6:
        st.markdown("<h2 class='sub-header'>Story with Dialogue</h2>", unsafe_allow_html=True)
        
        # Show tabs for episodes with dialogue
        if st.session_state.episodes:
            tab_titles = [f"Episode {ep.number}: {ep.title}" for ep in st.session_state.episodes]
            tabs = st.tabs(tab_titles)
            
            for i, tab in enumerate(tabs):
                with tab:
                    episode = st.session_state.episodes[i]
                    episode_num = episode.number
                    
                    # Display episode content
                    enhanced = st.session_state.enhanced_episodes.get(episode_num)
                    content = ""
                    if enhanced:
                        if hasattr(enhanced, 'lengthened_content'):
                            content = enhanced.lengthened_content
                        else:
                            content = enhanced.get('lengthened_content', '')
                    
                    st.markdown("<h3>Episode Content</h3>", unsafe_allow_html=True)
                    st.write(content)
                    
                    # Display dialogue
                    dialogue = st.session_state.dialogues.get(episode_num, "")
                    if dialogue:
                        st.markdown("<h3>Dialogue</h3>", unsafe_allow_html=True)
                        st.write(dialogue)
        
        # Finalize story button
        if st.button("Finalize Story"):
            with st.spinner("Finalizing story and saving files..."):
                story_data, story_dir = finalize_story(
                    st.session_state.topic,
                    st.session_state.story_type,
                    st.session_state.languages
                )
            
            st.success("Story generation complete!")
            st.session_state.current_step = 7
            st.rerun()
    
    # Step 7: Downloads
    elif st.session_state.current_step == 7:
        st.markdown("<h2 class='sub-header'>Final Story</h2>", unsafe_allow_html=True)
        
        st.success(f"Your story '{st.session_state.topic}' has been successfully generated!")
        
        # Show download options
        st.markdown("<h3>Download Options</h3>", unsafe_allow_html=True)
        
        if st.session_state.story_dir:
            # Complete story
            final_story_path = os.path.join(st.session_state.story_dir, "final_story.md")
            story_details_path = os.path.join(st.session_state.story_dir, "story_details.md")
            
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
            
            if os.path.exists(story_details_path):
                with open(story_details_path, 'r', encoding='utf-8') as f:
                    story_details_content = f.read()
                
                st.download_button(
                    label="Download Story Details (MD)",
                    data=story_details_content,
                    file_name="story_details.md",
                    mime="text/markdown",
                    key="download_story_details"
                )
            
            # Show translation downloads if available
            if st.session_state.translated_files:
                st.markdown("<h3>Translated Versions</h3>", unsafe_allow_html=True)
                
                for language, file_path in st.session_state.translated_files:
                    if os.path.exists(file_path):
                        with open(file_path, 'r', encoding='utf-8') as f:
                            translated_content = f.read()
                        
                        file_name = os.path.basename(file_path)
                        st.download_button(
                            label=f"Download {language} Translation",
                            data=translated_content,
                            file_name=file_name,
                            mime="text/markdown",
                            key=f"download_{language}"
                        )
        
        # Generate new story button
        if st.button("Generate New Story"):
            # Reset session state
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            init_session_state()
            st.rerun()

if __name__ == "__main__":
    main()
