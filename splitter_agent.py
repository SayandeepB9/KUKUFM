import os
from dotenv import load_dotenv  
load_dotenv()
from typing import List, Dict, Any
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from llm_api import llm_api, get_model_from_config
import json
import re

class Episode(BaseModel):
    number: int = Field(..., description="The sequential number of the episode.")
    title: str = Field(..., description="A short title capturing the main idea of the episode.")
    content: str = Field(..., description="A detailed narrative of the episode, developing and advancing the plot.")
    cliffhanger: str = Field("", description="For every episode except the final one, provide a dramatic cliffhanger. For the final episode, leave this empty.")
    
    def to_dict(self):
        """Return a dictionary representation of the episode for JSON serialization"""
        return {
            "number": self.number,
            "title": self.title,
            "content": self.content,
            "cliffhanger": self.cliffhanger
        }

class SplitStoryQuery(BaseModel):
    outline: str = Field(..., description="A general overview of the story's theme and direction.")
    characters: List[dict] = Field(
        ..., 
        description="A list of character objects. Each object should contain at least 'name', 'description', and 'role'."
    )
    num_episodes: int = Field(5, description="The total number of episodes to divide the story into.")
    episodes: List[Episode] = Field(default_factory=list, description="The generated episodes that model the long story.")
    
    class Config:
        # This is important for OpenAI models
        extra = "forbid"

class StorySplitterAgent:
    def __init__(self, api_key=None):
        # Get the model type and name from config
        self.model_type = "story_splitting"
        self.model_name = get_model_from_config(self.model_type)
        self.llm = llm_api(api_key=api_key, model_type=self.model_type)
        
        self.system_prompt = (
            "You are a master storyteller. Your task is to split a long narrative into interconnected episodes. "
            "Given an overall outline, specific plot points, and detailed character descriptions, generate a long, engaging story divided into episodes. "
            "For each episode, create a descriptive title, a detailed content section that moves the story forward, "
            "and a dramatic cliffhanger that leaves the reader eager to continue, except for the final episode. "
            "The final episode should have an empty cliffhanger field."
        )
        
    def split_story(self, detailed_plot: str, characters: List[dict], num_episodes: int = 5):
        """Generate episodes directly without structured output"""
        print(f"Generating {num_episodes} episodes for the story...")
        
        # Validate inputs to prevent errors
        if not detailed_plot or not isinstance(detailed_plot, str):
            detailed_plot = "A mysterious story with unexpected twists."
        
        if not characters or not isinstance(characters, list):
            characters = [
                {"name": "Character 1", "description": "A protagonist", "role": "Main character"},
                {"name": "Character 2", "description": "A supporting character", "role": "Friend"},
                {"name": "Character 3", "description": "An antagonist", "role": "Villain"}
            ]
        
        # Adapt the prompt based on whether we're using GPT or Groq
        if 'gpt' in self.model_name.lower():
            # GPT tends to work better with clear structure and examples
            direct_prompt = f"""
            Create exactly {num_episodes} sequential episodes for a story with the following details:
            
            DETAILED PLOT: {detailed_plot}
            
            CHARACTERS: {json.dumps([{"name": c["name"], "role": c["role"]} for c in characters])}
            
            For each episode, follow this specific format:
            
            EPISODE NUMBER: [number]
            TITLE: [title]
            CONTENT: [detailed content - at least 200 words]
            CLIFFHANGER: [cliffhanger - should be empty for the final episode]
            
            Example formatting (do not use this content, just follow the format):
            
            EPISODE NUMBER: 1
            TITLE: The Beginning
            CONTENT: This is where the detailed narrative goes...
            CLIFFHANGER: This is where you include a dramatic ending to keep readers engaged.
            
            EPISODE NUMBER: 2
            TITLE: The Middle
            ...and so on.
            
            Make sure to create exactly {num_episodes} episodes that tell a complete story.
            """
        else:
            # Groq models sometimes need simpler instructions
            direct_prompt = f"""
            Split the following story into {num_episodes} episodes:
            
            Story plot: {detailed_plot}
            
            Main characters: {", ".join([c["name"] + " (" + c["role"] + ")" for c in characters])}
            
            For each episode, please provide:
            1. Episode number (1 through {num_episodes})
            2. A title
            3. Detailed content (the story narrative)
            4. A cliffhanger (except for the final episode)
            
            Use this exact format:
            EPISODE NUMBER: (number)
            TITLE: (title)
            CONTENT: (content)
            CLIFFHANGER: (cliffhanger)
            
            For the final episode (Episode {num_episodes}), leave the CLIFFHANGER field empty.
            """
        
        # Get the response
        print(f"Generating episodes using {self.model_name}...")
        response = self.llm.invoke(direct_prompt)
        content = response.content if hasattr(response, 'content') else str(response)
        
        # Parse the response
        episodes = self._parse_episodes(content, num_episodes, detailed_plot)
        
        # Return the episodes
        return episodes
    
    def _parse_episodes(self, content, num_episodes, outline):
        """Parse episodes from various response formats"""
        episodes = []
        
        # Try multiple parsing approaches
        parsing_approaches = [
            self._parse_with_specific_markers,
            self._parse_with_episode_splitter,
            self._parse_with_number_title_pattern,
            self._parse_with_loose_pattern
        ]
        
        for approach_func in parsing_approaches:
            try:
                parsed_episodes = approach_func(content, num_episodes)
                if parsed_episodes and len(parsed_episodes) > 0:
                    print(f"Successfully parsed {len(parsed_episodes)} episodes")
                    episodes = parsed_episodes
                    break
            except Exception as e:
                print(f"Parsing approach failed: {str(e)}")
                continue
        
        # If all parsing attempts failed, create generic episodes
        if not episodes:
            print("All parsing attempts failed, creating generic episodes")
            for i in range(1, num_episodes + 1):
                episode = Episode(
                    number=i,
                    title=f"Episode {i}",
                    content=f"Content for episode {i} of the story about {outline[:50]}...",
                    cliffhanger="" if i == num_episodes else f"Cliffhanger for episode {i}"
                )
                episodes.append(episode)
        
        # Sort episodes by number to ensure correct order
        episodes.sort(key=lambda x: x.number)
        
        # Validate episode count
        if len(episodes) < num_episodes:
            print(f"Only parsed {len(episodes)} episodes, creating additional ones")
            current_numbers = [e.number for e in episodes]
            for i in range(1, num_episodes + 1):
                if i not in current_numbers:
                    episode = Episode(
                        number=i,
                        title=f"Episode {i}",
                        content=f"Content for episode {i} of the story about {outline[:50]}...",
                        cliffhanger="" if i == num_episodes else f"Cliffhanger for episode {i}"
                    )
                    episodes.append(episode)
            
            # Re-sort after adding
            episodes.sort(key=lambda x: x.number)
        
        # Ensure no cliffhanger on final episode
        for episode in episodes:
            if episode.number == num_episodes:
                episode.cliffhanger = ""
        
        return episodes
    
    def _parse_with_specific_markers(self, content, num_episodes):
        """Parse episodes using specific marker format - works well with GPT"""
        episodes = []
        
        # This pattern looks for clearly delineated episode sections
        pattern = r"EPISODE\s*NUMBER:\s*(\d+)[^\n]*\n+\s*TITLE:\s*([^\n]*)\n+\s*CONTENT:\s*((?:(?!EPISODE\s*NUMBER:|TITLE:|CLIFFHANGER:).)*)\s*CLIFFHANGER:\s*((?:(?!EPISODE\s*NUMBER:).)*)"
        matches = re.findall(pattern, content, re.DOTALL)
        
        if not matches or len(matches) == 0:
            return []
        
        for match in matches:
            try:
                number = int(match[0].strip())
                title = match[1].strip()
                content_text = match[2].strip()
                cliffhanger = match[3].strip() if number != num_episodes else ""
                
                episode = Episode(
                    number=number,
                    title=title,
                    content=content_text,
                    cliffhanger=cliffhanger
                )
                episodes.append(episode)
            except Exception as e:
                print(f"Error parsing episode: {str(e)}")
        
        return episodes
    
    def _parse_with_episode_splitter(self, content, num_episodes):
        """Parse by splitting on 'EPISODE NUMBER' markers - works with various formats"""
        episodes = []
        
        sections = content.split("EPISODE NUMBER:")
        if len(sections) <= 1:
            sections = content.split("Episode ")
            if len(sections) <= 1:
                return []
        
        for i, section in enumerate(sections[1:], 1):  # Skip the first split which is before any marker
            try:
                # Try to extract the episode number
                number_match = re.search(r"^\s*:?\s*(\d+)", section)
                if number_match:
                    number = int(number_match.group(1))
                else:
                    number = i
                
                # Extract title - look for explicit TITLE: marker or a line ending with a colon
                title_match = re.search(r"TITLE:\s*([^\n]+)", section)
                if not title_match:
                    title_match = re.search(r"(.+?):", section)
                
                title = title_match.group(1).strip() if title_match else f"Episode {number}"
                
                # Extract content - everything between CONTENT: and CLIFFHANGER:
                content_match = re.search(r"CONTENT:\s*((?:(?!CLIFFHANGER:).)*)", section, re.DOTALL)
                if not content_match:
                    # If no CONTENT marker, look for content after the title
                    content_text = re.sub(r"^.*?\n", "", section, 1)  # Remove the first line (with number/title)
                    content_text = re.sub(r"CLIFFHANGER:.*", "", content_text, flags=re.DOTALL)  # Remove cliffhanger
                else:
                    content_text = content_match.group(1)
                
                # Clean content
                content_text = content_text.strip()
                
                # Extract cliffhanger
                cliffhanger_match = re.search(r"CLIFFHANGER:\s*(.*?)($|\n\n)", section, re.DOTALL)
                cliffhanger = cliffhanger_match.group(1).strip() if cliffhanger_match else ""
                
                # Create the episode
                if content_text:  # Only add if we have content
                    episode = Episode(
                        number=number,
                        title=title,
                        content=content_text,
                        cliffhanger="" if number == num_episodes else cliffhanger
                    )
                    episodes.append(episode)
            except Exception as e:
                print(f"Error parsing episode {i}: {str(e)}")
        
        return episodes
    
    def _parse_with_number_title_pattern(self, content, num_episodes):
        """Parse using episode number and title patterns - good for Groq"""
        episodes = []
        
        # Look for patterns like "Episode 1: Title" or "Chapter 1 - Title"
        pattern = r"(?:Episode|Chapter)\s+(\d+)[:\s-]+([^\n]+)(?:\n|\r\n?)((?:(?!(?:Episode|Chapter)\s+\d+).)*)"
        matches = re.findall(pattern, content, re.DOTALL)
        
        if not matches:
            return []
        
        for i, match in enumerate(matches):
            try:
                number = int(match[0])
                title = match[1].strip()
                full_content = match[2].strip()
                
                # Try to separate content from cliffhanger
                cliffhanger = ""
                content_text = full_content
                
                # Look for cliffhanger indicators
                cliffhanger_indicators = [
                    "Cliffhanger:", "CLIFFHANGER:", 
                    "To be continued", "To Be Continued",
                    "Suddenly,", "Just then,", "At that moment,"
                ]
                
                # Check for explicit cliffhanger section
                for indicator in cliffhanger_indicators:
                    if indicator in full_content:
                        parts = full_content.split(indicator, 1)
                        if len(parts) == 2:
                            content_text = parts[0].strip()
                            cliffhanger = (indicator + " " + parts[1]).strip()
                            break
                
                # If this is not the last episode and no cliffhanger found, 
                # take the last paragraph as a cliffhanger
                if number != num_episodes and not cliffhanger:
                    paragraphs = full_content.split("\n\n")
                    if len(paragraphs) > 1:
                        content_text = "\n\n".join(paragraphs[:-1])
                        cliffhanger = paragraphs[-1]
                
                # Don't allow cliffhanger on final episode
                if number == num_episodes:
                    cliffhanger = ""
                
                episode = Episode(
                    number=number,
                    title=title,
                    content=content_text,
                    cliffhanger=cliffhanger
                )
                episodes.append(episode)
            except Exception as e:
                print(f"Error parsing episode with pattern: {str(e)}")
        
        return episodes
    
    def _parse_with_loose_pattern(self, content, num_episodes):
        """Fallback parser for less structured content"""
        episodes = []
        
        # Split content into chunks that might be episodes
        chunks = re.split(r'\n\n\n+|\r\n\r\n\r\n+', content)
        
        # If we have too few chunks, try splitting by double newlines
        if len(chunks) < num_episodes:
            chunks = re.split(r'\n\n|\r\n\r\n', content)
        
        # Filter out empty chunks
        chunks = [chunk for chunk in chunks if chunk.strip()]
        
        # Try to identify episodes from these chunks
        current_episode = None
        
        for i, chunk in enumerate(chunks):
            # Check if this chunk starts a new episode
            episode_start = re.search(r'(?:episode|chapter|part)\s*(\d+)', chunk.lower())
            
            if episode_start:
                # If we already have an episode in progress, save it
                if current_episode:
                    episodes.append(current_episode)
                
                # Start a new episode
                number = int(episode_start.group(1))
                
                # Try to extract a title
                title_match = re.search(r'(?:episode|chapter|part)\s*\d+\s*:?\s*(.+?)[\n\r]', chunk.lower())
                title = title_match.group(1).strip() if title_match else f"Episode {number}"
                
                # Remove the episode header from content
                content_text = re.sub(r'^.*?[\n\r]', '', chunk, 1, re.IGNORECASE)
                
                # Create the episode
                current_episode = Episode(
                    number=number,
                    title=title,
                    content=content_text,
                    cliffhanger="" if number == num_episodes else ""
                )
            elif current_episode:
                # Add this chunk to the current episode's content
                current_episode.content += "\n\n" + chunk
        
        # Add the last episode if we have one
        if current_episode:
            episodes.append(current_episode)
        
        # If we have enough episodes, try to assign cliffhangers
        if len(episodes) >= 2:
            for i in range(len(episodes) - 1):  # Skip the last episode
                content = episodes[i].content
                paragraphs = content.split("\n\n")
                
                if len(paragraphs) > 1:
                    episodes[i].content = "\n\n".join(paragraphs[:-1])
                    episodes[i].cliffhanger = paragraphs[-1]
        
        return episodes


if __name__ == "__main__":
    # Test the splitter with sample data
    splitter = StorySplitterAgent()
    outline_text = "A mysterious struggle between light and dark unfolds in a haunted city where supernatural forces battle unseen foes."
    characters_list = [
        {"name": "Alice", "description": "A determined detective haunted by her past, driven by justice.", "role": "Protagonist"},
        {"name": "Victor", "description": "A conflicted priest caught between faith and duty, harboring deep secrets.", "role": "Supporting"},
        {"name": "Evelyn", "description": "A spectral figure with an enigmatic past, whose guidance is as eerie as it is pivotal.", "role": "Antagonist/Guide"}
    ]
    episodes = splitter.split_story(outline_text, characters_list, num_episodes=5)
    print("Generated Episodes:", episodes)