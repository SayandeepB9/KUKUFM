from pathlib import Path
from openai import OpenAI


class AudioGenerator:
    def __init__(self, api_key=None):
        self.client = OpenAI(api_key=api_key)
        
    def generate_audio(self, text: str, voice: str = "coral", output_path: str = None):
        """
        Generate audio from text using OpenAI's TTS API.
        
        Args:
            text (str): The text to convert to speech
            voice (str): The voice to use (default: "coral")
            output_path (str): Path to save the audio file
            
        Returns:
            Path: Path to the generated audio file
        """
        if output_path is None:
            output_path = Path(__file__).parent / "speech.mp3"
        else:
            output_path = Path(output_path)
            
        try:
            with self.client.audio.speech.with_streaming_response.create(
                model="tts-1",
                voice=voice,
                input=text,
                instructions="Speak in a natural and engaging tone."
            ) as response:
                response.stream_to_file(output_path)
                
            return output_path
            
        except Exception as e:
            print(f"Error generating audio: {str(e)}")
            return None

if __name__ == "__main__":
    # Example usage
    generator = AudioGenerator()
    text = "Today is a wonderful day to build something people love!"
    audio_path = generator.generate_audio(text)
    if audio_path:
        print(f"Audio generated successfully at: {audio_path}")
    else:
        print("Failed to generate audio") 