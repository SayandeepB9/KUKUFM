import asyncio
import os
import json
from pathlib import Path
from typing import Optional, List, Dict, Any
from openai import AsyncOpenAI, OpenAI
from openai.helpers import LocalAudioPlayer
from dotenv import load_dotenv
import streamlit as st

# Load environment variables
load_dotenv()

class TextToSpeechAgent:
    """Agent for converting text to speech using OpenAI's TTS API."""
    
    def __init__(self, api_key=None):
        """
        Initialize the text-to-speech agent.
        
        Args:
            api_key (str, optional): API key for OpenAI. If not provided, it will be loaded from environment.
        """
        # Load API key from environment if not provided
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key not found. Please set OPENAI_API_KEY environment variable.")
            
        # Initialize OpenAI clients (sync and async)
        self.sync_client = OpenAI(api_key=self.api_key)
        self.async_client = AsyncOpenAI(api_key=self.api_key)
        
        # Default TTS parameters
        self.model = "gpt-4o-mini-tts"  # Use tts-1 for higher quality or tts-1-hd for highest quality
        self.voice = "nova"   # Options: alloy, echo, fable, onyx, nova, shimmer
    
    async def stream_audio(self, text: str, voice: str = None, instructions: str = "Speak in a natural and engaging tone.") -> None:
        """
        Stream audio directly to speakers using OpenAI's streaming API.
        
        Args:
            text (str): The text to convert to speech
            voice (str, optional): Voice to use. Defaults to the instance's default voice.
            instructions (str, optional): Instructions for the speech style.
        """
        voice_to_use = voice or self.voice
        
        try:
            # Truncate very long text to avoid API limits
            max_chars = 4000  # OpenAI's limit is around 4096 chars
            if len(text) > max_chars:
                print(f"Text too long ({len(text)} chars), truncating to {max_chars} chars")
                text = text[:max_chars] + "..."
            
            print(f"Streaming audio with voice: {voice_to_use}")
            async with self.async_client.audio.speech.with_streaming_response.create(
                model=self.model,
                voice=voice_to_use,
                input=text,
                instructions=instructions,
                response_format="mp3",  # or "pcm" for raw audio
            ) as response:
                await LocalAudioPlayer().play(response)
                
            print("Audio streaming completed")
            return True
            
        except Exception as e:
            print(f"Error streaming audio: {str(e)}")
            return False
    
    def generate_speech(self, text: str, output_path: str = None, voice: str = None) -> Optional[str]:
        """
        Generate audio file from text using OpenAI's TTS API.
        
        Args:
            text (str): The text to convert to speech
            output_path (str, optional): Path to save the audio file. Defaults to 'speech.mp3'.
            voice (str, optional): Voice to use. Defaults to the instance's default voice.
            
        Returns:
            str: Path to the generated audio file or None on failure
        """
        if output_path is None:
            output_path = "speech.mp3"
            
        voice_to_use = voice or self.voice
            
        try:
            # Truncate very long text to avoid API limits
            max_chars = 4000  # OpenAI's limit is around 4096 chars
            if len(text) > max_chars:
                print(f"Text too long ({len(text)} chars), truncating to {max_chars} chars")
                text = text[:max_chars] + "..."
            
            print(f"Generating audio with voice: {voice_to_use}")
            response = self.sync_client.audio.speech.create(
                model=self.model,
                voice=voice_to_use,
                input=text,
                response_format="mp3"
            )
            
            # Ensure directory exists
            os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
            
            # Save audio to file
            response.stream_to_file(output_path)
            print(f"Audio saved to: {output_path}")
            
            return output_path
        except Exception as e:
            print(f"Error generating speech: {str(e)}")
            return None
    
    def generate_episode_audio(self, episode_content: str, episode_number: int, output_dir: str) -> Optional[str]:
        """
        Generate audio for a specific episode and save it to the specified directory.
        
        Args:
            episode_content (str): The text content of the episode
            episode_number (int): The episode number
            output_dir (str): Directory to save the audio files
            
        Returns:
            str: Path to the generated audio file
        """
        # Create audio directory if it doesn't exist
        audio_dir = os.path.join(output_dir, "audio")
        os.makedirs(audio_dir, exist_ok=True)
        
        # Create output path for this episode
        output_path = os.path.join(audio_dir, f"episode_{episode_number}.mp3")
        
        # Truncate very long episodes to avoid API limits
        max_chars = 4000  # OpenAI typically has character limits for TTS requests
        if len(episode_content) > max_chars:
            print(f"Episode content too long ({len(episode_content)} chars), truncating to {max_chars} chars")
            episode_content = episode_content[:max_chars] + "..."
        
        # Generate audio
        return self.generate_speech(episode_content, output_path)
    
    async def play_episode_audio(self, episode_content: str, voice: str = None) -> bool:
        """
        Play episode audio directly using streaming API.
        
        Args:
            episode_content (str): The text content of the episode
            voice (str, optional): Voice to use. Defaults to the instance's default voice.
            
        Returns:
            bool: True if successful, False otherwise
        """
        return await self.stream_audio(episode_content, voice)


# Example usage
if __name__ == "__main__":
    # Test the streaming functionality
    async def test_streaming():
        agent = TextToSpeechAgent()
        text = "Today is a wonderful day to build something people love!"
        await agent.stream_audio(text, voice="nova")
    
    # Test the file generation functionality
    def test_file_generation():
        agent = TextToSpeechAgent()
        text = "Today is a wonderful day to build something people love!"
        audio_path = agent.generate_speech(text)
        if audio_path:
            print(f"Audio generated successfully at: {audio_path}")
        else:
            print("Failed to generate audio")
    
    # Run the async test
    asyncio.run(test_streaming())
    
    # Run the sync test
    test_file_generation()


