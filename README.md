# KUKUFM Story Generation Framework

A comprehensive framework for AI-powered story generation, character development, and plot enhancement.

## Overview

This project provides a set of tools for creative writing using Large Language Models (LLMs). It includes:

- Story outline generation
- Character development
- Plot options and twists
- Integrated LLM API support for both OpenAI and Groq models

## Setup

1. Clone the repository
2. Install requirements:
   ```bash
   pip install -r requirements.txt
   ```
3. Create a `.env` file with your API keys:
   ```
   OPENAI_API_KEY=your_openai_api_key_here
   GROQ_API_KEY=your_groq_api_key_here
   ```

## Usage

### LLM API

The `llm_api.py` module provides a unified interface to both OpenAI and Groq models:

```python
from llm_api import llm_api

# Initialize an OpenAI model
openai_llm = llm_api(model="gpt-4o-mini")

# Initialize a Groq model
groq_llm = llm_api(model="llama3-70b-8192")
```

### Story Generation

```python
from outline_generation_agent import OutlineGenerator
from character_development_agent import CharacterDevelopmentAgent
from plot_selector import StoryElementLibrary

# Generate a story outline
generator = OutlineGenerator(model="llama3-70b-8192", api_key=None)
outline = generator.generate_outline("A mystery in an ancient temple")

# Generate characters for the story
character_agent = CharacterDevelopmentAgent(model="llama3-70b-8192", api_key=None)
characters = character_agent.generate_characters("A mystery in an ancient temple")

# Generate plot options
library = StoryElementLibrary(model_name="llama3-70b-8192")
plot_options = library.generate_plot_options(outline, "mystery")
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.
