from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
import os
import yaml
from openai import OpenAI
from dotenv import load_dotenv
import sys

# Load environment variables from .env file
load_dotenv()

# Load configuration from YAML file
config_path = os.path.join(os.path.dirname(__file__), 'config.yaml')
try:
    with open(config_path, 'r') as file:
        print(f"Loading config from {config_path}")
        config = yaml.safe_load(file)
        print(f"Config loaded successfully: {list(config.keys())}")
except Exception as e:
    print(f"Error loading config file from {config_path}: {e}")
    print("Using default configuration instead")
    config = {
        "models": {"default": "llama3-70b-8192"},
        "api_keys": {"groq": "", "openai": ""},
        "settings": {"temperature": 0.7, "streaming": False}
    }

# Get API keys from environment variables (with fallback to config file)
openai_api_key = os.getenv('OPENAI_API_KEY') or config['api_keys'].get('openai', '')
groq_api_key = os.getenv('GROQ_API_KEY') or config['api_keys'].get('groq', '')

if not groq_api_key:
    print("WARNING: No Groq API key found in environment variables or config!")
    print("Set GROQ_API_KEY in your .env file or provide it directly when calling functions.")

# Initialize OpenAI client with API key
try:
    if openai_api_key:
        client = OpenAI(api_key=openai_api_key)
        print("OpenAI client initialized successfully")
    else:
        print("WARNING: No OpenAI API key found, OpenAI client not initialized")
        client = None
except Exception as e:
    print(f"ERROR initializing OpenAI client: {e}")
    client = None

def get_model_from_config(model_type="default"):
    """
    Get model name from config based on model type
    """
    model_name = config['models'].get(model_type, config['models'].get('default'))
    print(f"Selected model for {model_type}: {model_name}")
    return model_name

def get_settings_from_config():
    """
    Get default settings from config
    """
    settings = config.get('settings', {"temperature": 0.7, "streaming": False})
    print(f"Using settings: {settings}")
    return settings

def llm_api(model=None, api_key=None, temperature=None, streaming=None, model_type="default"):
    """
    Returns an LLM instance based on the specified model or model type from config.
    
    Args:
        model (str, optional): The model to use (e.g., "gpt-4o-mini", "llama3-70b-8192")
                               If None, gets model from config based on model_type
        model_type (str): Type of model to use from config (e.g., "default", "outline_generation")
        api_key (str, optional): API key for the model provider. If None, uses environment variables.
        temperature (float, optional): Temperature setting for the model. Uses config if None.
        streaming (bool, optional): Whether to enable streaming. Uses config if None.
    
    Returns:
        LLM instance compatible with LangChain
    """
    print(f"\nInitializing LLM for {model_type} task")
    
    # Get model from config if not specified
    if model is None:
        model = get_model_from_config(model_type)
    else:
        print(f"Using explicitly provided model: {model}")
    
    # Get settings from config if not specified
    settings = get_settings_from_config()
    if temperature is None:
        temperature = settings.get("temperature", 0.7)
    if streaming is None:
        streaming = settings.get("streaming", False)
    
    print(f"LLM settings - Model: {model}, Temperature: {temperature}, Streaming: {streaming}")
    
    try:
        if "gpt" in model.lower():
            # Use the loaded OpenAI API key if none is provided
            if api_key is None:
                api_key = openai_api_key
                print("Using OpenAI API key from environment/config")
            
            if not api_key:
                print("ERROR: No OpenAI API key available!")
                return None
            
            # Return LangChain's ChatOpenAI instance
            print("Initializing ChatOpenAI...")
            llm = ChatOpenAI(
                model=model,
                openai_api_key=api_key,
                temperature=temperature,
                streaming=streaming
            )
            print("ChatOpenAI initialized successfully")
            return llm
        
        else:  # Assume it's a Groq model
            # Use the loaded Groq API key if none is provided
            if api_key is None:
                api_key = groq_api_key
                print("Using Groq API key from environment/config")
                
            if not api_key:
                print("ERROR: No Groq API key available!")
                return None
                
            # If model name doesn't specify a provider, default to llama3
            if "llama" not in model.lower() and "claude" not in model.lower():
                model = get_model_from_config("default")
                print(f"Using default model: {model}")
                
            # Return LangChain's ChatGroq instance
            print("Initializing ChatGroq...")
            llm = ChatGroq(
                model_name=model,
                api_key=api_key,
                temperature=temperature,
                streaming=streaming
            )
            print("ChatGroq initialized successfully")
            return llm
            
    except Exception as e:
        print(f"ERROR initializing LLM: {e}")
        print(f"Model: {model}, Model Type: {model_type}")
        return None