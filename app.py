import streamlit as st
import os
from utils import (
    initialize_model,
    generate_response,
    count_tokens,
    save_chat_history,
    load_chat_history
)

# Application configuration
st.set_page_config(
    page_title="AI Chat Interface",
    page_icon=":speech_balloon:",
    layout="wide"
)

# Session state initialization
if 'chats' not in st.session_state:
    st.session_state.chats = []

if 'current_chat' not in st.session_state:
    st.session_state.current_chat = None

if 'model_type' not in st.session_state:
    st.session_state.model_type = "azure"

def create_new_chat():
    """Create a new chat session."""
    new_chat = {
        'id': len(st.session_state.chats) + 1,
        'name': f"Chat {len(st.session_state.chats) + 1}",
        'messages': []
    }
    st.session_state.chats.append(new_chat)
    st.session_state.current_chat = new_chat

def delete_chat(chat_to_delete):
    """Delete a specific chat session."""
    st.session_state.chats = [
        chat for chat in st.session_state.chats 
        if chat['id'] != chat_to_delete['id']
    ]
    
    if st.session_state.chats:
        st.session_state.current_chat = st.session_state.chats[-1]
    else:
        create_new_chat()

def main():
    # Sidebar for chat management
    with st.sidebar:
        st.title("AI Chat Interface")
        
        # Model selection
        st.session_state.model_type = st.radio(
            "Select Model Platform",
            ["azure", "claude"]
        )
        
        if st.session_state.model_type == "azure":
            deployment_name = st.selectbox(
                "Select Azure Model",
                ["gpt-4", "gpt-3.5-turbo", "gpt-4-turbo"]
            )
        else:
            deployment_name = "claude-3-sonnet"
        
        # Create new chat button
        if st.button("➕ New Chat"):
            create_new_chat()
        
        # Chat selection
        st.write("### Your Chats")
        for chat in st.session_state.chats:
            col1, col2 = st.columns([3, 1])
            with col1:
                if st.button(chat['name'], key=f"chat_{chat['id']}"):
                    st.session_state.current_chat = chat
            with col2:
                if st.button("🗑️", key=f"delete_{chat['id']}"):
                    delete_chat(chat)
        
        max_tokens = st.slider("Max Response Tokens", 100, 16000, 5000)

    # Initialize selected model client
    client = initialize_model(st.session_state.model_type)
    if not client:
        st.error(f"Failed to initialize {st.session_state.model_type} client. Check your credentials.")
        return

    # Main chat interface
    if not st.session_state.chats:
        create_new_chat()

    current_chat = st.session_state.current_chat
    st.header(current_chat['name'])

    # Display chat messages
    for msg in current_chat['messages']:
        with st.chat_message(msg['role']):
            st.markdown(msg['content'])

    # Chat input
    if prompt := st.chat_input("Enter your message"):
        # Add user message to chat history
        current_chat['messages'].append({
            'role': 'user',
            'content': prompt
        })

        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate AI response
        with st.chat_message("assistant"):
            with st.spinner("Generating response..."):
                # Prepare messages for API call
                messages = [
                    {"role": m['role'], "content": m['content']}
                    for m in current_chat['messages']
                ]

                # Generate response
                response = generate_response(
                    client,
                    messages,
                    model_type=st.session_state.model_type,
                    deployment_name=deployment_name
                )

                # Display and save response
                st.markdown(response)
                
                # Add AI response to chat history
                current_chat['messages'].append({
                    'role': 'assistant',
                    'content': response
                })

    # Token usage display
    total_tokens = sum(
        count_tokens(
            msg['content'],
            model=deployment_name
        )
        for msg in current_chat['messages']
    )
    st.sidebar.info(f"Total Tokens Used: {total_tokens}")

    # Save chat history option
    if st.sidebar.button("💾 Save Chat History"):
        save_chat_history(current_chat['messages'])

if __name__ == "__main__":
    main()