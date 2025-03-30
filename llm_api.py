from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
import os
import yaml
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Load configuration from YAML file
config_path = os.path.join(os.path.dirname(__file__), 'config.yaml')
try:
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
except Exception as e:
    print(f"Error loading config file: {e}")
    config = {
        "models": {"default": "llama3-70b-8192"},
        "api_keys": {"groq": "", "openai": ""},
        "settings": {"temperature": 0.7, "streaming": False}
    }

# Get API keys from environment variables (with fallback to config file)
openai_api_key = os.getenv('OPENAI_API_KEY') or config['api_keys'].get('openai', '')
groq_api_key = os.getenv('GROQ_API_KEY') or config['api_keys'].get('groq', '')

# Initialize OpenAI client with API key
try:
    if openai_api_key:
        client = OpenAI(api_key=openai_api_key)
    else:
        client = None
except Exception as e:
    print(f"ERROR initializing OpenAI client: {e}")
    client = None

def get_model_from_config(model_type="default"):
    """
    Get model name from config based on model type
    """
    # If provider-specific model type is specified, use that
    if model_type.startswith('openai_'):
        return config['models'].get(model_type, config['models'].get('openai_default', 'gpt-4o-mini'))
    
    # Otherwise get the model from the model configuration
    model_name = config['models'].get(model_type, config['models'].get('default', 'llama3-70b-8192'))
    return model_name

def get_settings_from_config():
    """
    Get default settings from config
    """
    return config.get('settings', {"temperature": 0.7, "streaming": False})

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
    # Get model from config if not specified
    if model is None:
        model = get_model_from_config(model_type)
    
    # Get settings from config if not specified
    settings = get_settings_from_config()
    if temperature is None:
        temperature = settings.get("temperature", 0.7)
    if streaming is None:
        streaming = settings.get("streaming", False)
    
    try:
        if "gpt" in model.lower():
            # Use the loaded OpenAI API key if none is provided
            if api_key is None:
                api_key = openai_api_key
            
            if not api_key:
                print("ERROR: No OpenAI API key available!")
                return None
            
            # Return LangChain's ChatOpenAI instance
            return ChatOpenAI(
                model=model,
                openai_api_key=api_key,
                temperature=temperature,
                streaming=streaming
            )
        
        else:  # Assume it's a Groq model
            # Use the loaded Groq API key if none is provided
            if api_key is None:
                api_key = groq_api_key
                
            if not api_key:
                print("ERROR: No Groq API key available!")
                return None
                
            # If model name doesn't specify a provider, default to llama3
            if "llama" not in model.lower() and "claude" not in model.lower():
                model = get_model_from_config("default")
                
            # Return LangChain's ChatGroq instance
            return ChatGroq(
                model_name=model,
                api_key=api_key,
                temperature=temperature,
                streaming=streaming
            )
            
    except Exception as e:
        print(f"ERROR initializing LLM: {e}")
        return None