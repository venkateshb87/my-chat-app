import os
import json
import tiktoken
import streamlit as st
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
from langchain.schema import HumanMessage, SystemMessage, AIMessage
from langchain_community.chat_models import BedrockChat
import boto3

__all__ = [
    'initialize_model',
    'generate_response',
    'count_tokens',
    'save_chat_history', 
    'load_chat_history'
]


load_dotenv()

def initialize_model(model_type="azure"):
    """
    Initialize the specified model client (Azure OpenAI or Claude).
    
    Args:
        model_type (str): Either "azure" or "claude"
    """
    if model_type == "azure":
        endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        api_key = os.getenv("AZURE_OPENAI_KEY")
        
        if not endpoint or not api_key:
            st.error("Azure OpenAI credentials not found!")
            return None
        
        try:
            return AzureChatOpenAI(
                temperature=0,
                azure_endpoint=endpoint,
                api_key=api_key,
                openai_api_version="2023-05-15",
                azure_deployment=os.getenv("OPENAI_GPT4_DEPLOYMENT_NAME"),
            )
        except Exception as e:
            st.error(f"Error initializing Azure OpenAI client: {e}")
            return None
    
    elif model_type == "claude":
        try:
            bedrock = boto3.client('bedrock-runtime')
            return BedrockChat(
                model_id="amazon.titan-text-express-v1",
                client=bedrock,
                model_kwargs={"temperature": 0},
                streaming=True
            )
        except Exception as e:
            st.error(f"Error initializing Claude client: {e}")
            return None
    
    else:
        st.error(f"Unknown model type: {model_type}")
        return None

def generate_response(client, messages, model_type="azure", deployment_name=None):
    """
    Generate a response using the specified model.
    
    Args:
        client: Initialized LangChain client
        messages (list): List of message dictionaries
        model_type (str): Either "azure" or "claude"
        deployment_name (str, optional): Name of the Azure deployment
    """
    try:
        langchain_messages = []
        for msg in messages:
            content = msg.get('content', '')
            role = msg.get('role', '')
            
            if role == 'system':
                langchain_messages.append(SystemMessage(content=content))
            elif role == 'user':
                langchain_messages.append(HumanMessage(content=content))
            elif role == 'assistant':
                langchain_messages.append(AIMessage(content=content))
        
        response = client.invoke(langchain_messages)
        return response.content
    except Exception as e:
        st.error(f"Error generating response: {e}")
        return f"An error occurred: {e}"


def count_tokens(text, model="gpt-4"):
    """Count tokens for given text based on model type."""
    try:
        if "claude" in model.lower():
            # Approximate token count for Claude (4 characters ≈ 1 token)
            return len(text) // 4
        else:
            encoding = tiktoken.encoding_for_model(model)
            return len(encoding.encode(text))
    except Exception as e:
        st.warning(f"Token counting error: {e}")
        return 0

def save_chat_history(chat_history, filename="chat_history.json"):
    """Save chat history to a JSON file."""
    try:
        with open(filename, 'w') as f:
            json.dump(chat_history, f, indent=2)
        st.success(f"Chat history saved to {filename}")
    except Exception as e:
        st.error(f"Error saving chat history: {e}")

def load_chat_history(filename="chat_history.json"):
    """Load chat history from a JSON file."""
    try:
        if os.path.exists(filename):
            with open(filename, 'r') as f:
                return json.load(f)
    except Exception as e:
        st.error(f"Error loading chat history: {e}")
    return []