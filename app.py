import streamlit as st
import os
import free_ask_internet

# Page Configuration
st.set_page_config(page_title="FreeAskInternet", page_icon="🔍", layout="wide")

# Sidebar Settings
st.sidebar.title("Settings")
model_name = st.sidebar.text_input("Model Name", value="gpt3.5")
llm_base_url = st.sidebar.text_input("LLM Base URL", value="http://llm-freegpt35:3040/v1/")
llm_auth_token = st.sidebar.text_input("LLM Auth Token", value="", type="password")
use_custom_llm = st.sidebar.checkbox("Use Custom LLM", value=True)
enable_search = st.sidebar.checkbox("Enable Web Search", value=True, help="Disable to chat with LLM directly using context memory.")

# History Limit
history_limit = st.sidebar.number_input("History Context Limit", min_value=0, max_value=20, value=5)
os.environ["LLM_HISTORY_LIMIT"] = str(history_limit)

# Session State Initialization
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display Chat History
st.title("FreeAskInternet 💬")
st.markdown("### AI Search Assistant with Memory")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat Input
if prompt := st.chat_input("What would you like to know?"):
    # Add user message to state
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate Assistant Response
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        
        # Prepare history for context (excluding current prompt which is passed as query)
        # The history passed to ask_internet should be in format [{"role":..., "content":...}]
        # excluding the current user prompt because ask_internet constructs the final prompt
        history_context = st.session_state.messages[:-1]

        # Call the backend
        # We stream the response
        try:
            # Decide on search: 
            # If enable_search is True, we search. 
            # Ideally, we could add an LLM step to classify intent, but for now we trust the toggle/default.
            
            # Using custom logic from free_ask_internet
            generator = free_ask_internet.ask_internet(
                query=prompt,
                history=history_context,
                model=model_name,
                llm_auth_token=llm_auth_token,
                llm_base_url=llm_base_url,
                using_custom_llm=use_custom_llm,
                search_enabled=enable_search
            )

            for token in generator:
                full_response += token
                message_placeholder.markdown(full_response + "▌")
            
            message_placeholder.markdown(full_response)
        
        except Exception as e:
            st.error(f"An error occurred: {e}")
            full_response = f"Error: {e}"

    # Add assistant response to state
    st.session_state.messages.append({"role": "assistant", "content": full_response})

