from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
import os
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Get API keys from environment variables (now loaded from .env file)
openai_api_key = os.getenv('OPENAI_API_KEY')
groq_api_key = os.getenv('GROQ_API_KEY')

# Initialize OpenAI client with API key
client = OpenAI(api_key=openai_api_key)

def llm_api(model="gpt-4o-mini", api_key=None, temperature=0.7, streaming=False):
    """
    Returns an LLM instance based on the specified model.
    
    Args:
        model (str): The model to use (e.g., "gpt-4o-mini", "llama3-70b-8192")
        api_key (str, optional): API key for the model provider. If None, uses environment variables.
        temperature (float, optional): Temperature setting for the model. Default is 0.7.
        streaming (bool, optional): Whether to enable streaming. Default is False.
    
    Returns:
        LLM instance compatible with LangChain
    """
    if "gpt" in model.lower():
        # Use the loaded OpenAI API key if none is provided
        if api_key is None:
            api_key = openai_api_key
        
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
            
        # If model name doesn't specify a provider, default to llama3
        if "llama" not in model.lower() and "claude" not in model.lower():
            model = "llama3-70b-8192"
            
        # Return LangChain's ChatGroq instance
        return ChatGroq(
            model_name=model,
            api_key=api_key,
            temperature=temperature,
            streaming=streaming
        )