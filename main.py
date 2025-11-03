from backend.core import run_llm
import streamlit as st
import time
from typing import Set


st.header("Langchain Udemy Course - Documentation Helper Bot")
prompt = st.text_input("Prompt", placeholder="Enter your prompt here")

if (
    "chat_answers_history" not in st.session_state
    and "user_prompt_history" not in st.session_state
    and "chat_history" not in st.session_state
):
    st.session_state["chat_answers_history"] = []
    st.session_state["user_prompt_history"] = []
    st.session_state["chat_history"] = []

# Sidebar: User information
with st.sidebar:
    st.header("User")

    # Defaults only set once
    if "user_name" not in st.session_state:
        st.session_state["user_name"] = "Jane Doe"
    if "user_email" not in st.session_state:
        st.session_state["user_email"] = "jane.doe@example.com"

    uploaded_avatar = st.file_uploader("Profile picture", type=["png", "jpg", "jpeg"], accept_multiple_files=False)
    if uploaded_avatar is not None:
        st.session_state["user_avatar_bytes"] = uploaded_avatar.read()

    # Display avatar if available
    if "user_avatar_bytes" in st.session_state and st.session_state["user_avatar_bytes"]:
        st.image(st.session_state["user_avatar_bytes"], caption="Profile", use_container_width=True)
    else:
        st.markdown("_No profile picture uploaded_")

    st.session_state["user_name"] = st.text_input("Name", value=st.session_state["user_name"])  # type: ignore[assignment]
    st.session_state["user_email"] = st.text_input("Email", value=st.session_state["user_email"])  # type: ignore[assignment]


def create_sources_string(source_urls: Set[str]) -> str:
    if not source_urls:
        return ""
    sources_list = list(source_urls)
    sources_list.sort()
    sources_string = "Sources:\n"
    for i, source in enumerate(sources_list):
        sources_string += f"[{i+1}]. {source}\n"
    return sources_string


if prompt:
    with st.spinner("Generating response..."):
        generated_response = run_llm(query = prompt, chat_history = st.session_state["chat_history"])
        sources = set([doc.metadata['source'] for doc in generated_response['source_documents']])

        formatted_response = f"{generated_response['result']}\n {create_sources_string(sources)}"

        st.session_state["user_prompt_history"].append(prompt)
        st.session_state["chat_answers_history"].append(formatted_response)
        st.session_state["chat_history"].append(("human", prompt))
        st.session_state["chat_history"].append(("ai", generated_response['result']))

if st.session_state["chat_answers_history"]:
    for generated_response, user_quesry in zip(st.session_state["chat_answers_history"], st.session_state["user_prompt_history"]):
        st.chat_message("user").write(user_quesry)
        st.chat_message("assistant").write(generated_response)
