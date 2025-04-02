# KUKUFM AI Story Generator

An AI-powered platform that generates engaging stories with episodic structure, character development, and dialogue, with support for human feedback and multilingual output.

## Overview

This project leverages Large Language Models (LLMs) to create compelling narratives through a pipeline of specialized agents that handle different aspects of storytelling. The system produces complete stories with episodes, characters, and realistic dialogue, allowing for human feedback and translations to multiple languages.

## Features

- **Topic-Based Story Generation**: Create stories based on any topic or theme
- **Interactive Outline Refinement**: Provide feedback to refine story outlines
- **Character Development**: Automatic generation of detailed characters with descriptions 
- **Episodic Structure**: Split stories into cohesive episodes with cliffhangers
- **Enhanced Content**: Expand episodes with detailed narratives and engagement points
- **Dialogue Generation**: Create realistic dialogue between characters
- **Multilingual Support**: Translate stories to different languages
- **Human Feedback Loop**: Iteratively improve outlines based on human input

## Components

The system consists of several specialized agents:

- **OutlineGenerator**: Creates the initial story outline and refines it based on feedback
- **CharacterDevelopmentAgent**: Develops detailed characters for the story
- **StoryElementLibrary**: Suggests plot elements and story devices
- **StorySplitterAgent**: Divides the narrative into coherent episodes
- **EpisodeLengtheningAgent**: Expands episode outlines into detailed content
- **DialogueAgent**: Generates natural dialogue between characters
- **TranslatorAgent**: Translates stories to different languages

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/kukufm-story-generator.git
   cd kukufm-story-generator
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Set up your environment variables in a `.env` file:
   ```
   OPENAI_API_KEY=your_openai_api_key
   OPENAI_API_MODEL=gpt-4o-mini  # or your preferred model
   ```

## Usage

Run the main script with a topic to generate a story:
