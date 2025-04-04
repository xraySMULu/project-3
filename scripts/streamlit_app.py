# Dungeon Master Project

import os
import sys
import streamlit as st
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
import json

# Add the path to our game logic module
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "code", "app")))

from game_state import (
    load_state,
    update_inventory,
    update_location,
    add_to_history,
    generate_prompt,
    save_state
)

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Initialize the model client
client = ChatOpenAI(
    model_name="gpt-3.5-turbo",
    temperature=0.9,  # Slight creativity
    api_key=OPENAI_API_KEY
)

# Basic Streamlit page setup
st.set_page_config("AI Dungeon Master")
st.title("AI Dungeon Master: Create Your Own Adventure")

# Sidebar button to start fresh
if st.sidebar.button("Start New Story"):
    fresh_state = {
        "location": "Starting Point",
        "inventory": [],
        "visited_locations": [],
        "history": [],
        "story_summary": "",
        "character_info": ""
    }
    save_state(fresh_state)
    st.rerun()

# Load existing state or default
state = load_state()

# If we're just starting, collect character details
if not state.get("history"):
    st.subheader("Create Your Character")

    name = st.text_input("Character Name")
    origin = st.text_input("Hometown")
    background = st.text_area("Backstory")
    starting_location = st.text_input("Starting Location", "Mysterious Forest")
    starting_scene = st.text_area(
        "Opening Scene Description",
        "You stand at the edge of a mysterious forest said to hold the key to an ancient power."
    )

    if st.button("Begin Adventure"):
        character_info = f"{name}, from {origin}. {background}"
        state["character_info"] = character_info

        # Assemble the intro prompt
        intro = f"You are {character_info}\n\n{starting_scene}\nWhat do you do?"
        state["location"] = starting_location
        state["visited_locations"].append(starting_location)

        # Call the model with the intro
        response = client.invoke(intro)
        dm_text = response.content

        add_to_history(state, intro, dm_text)
        save_state(state)
        st.rerun()
else:
    # Show the current game state info
    st.markdown(f"**📍 Location:** `{state['location']}`")
    st.markdown(f"**🎒 Inventory:** `{', '.join(state['inventory']) or 'Empty'}`")
    st.markdown(f"**🗺️ Visited:** `{', '.join(state['visited_locations'])}`")

    with st.form("player_input_form"):
        player_action = st.text_input("Your Action")
        submitted = st.form_submit_button("Submit")

    if submitted and player_action:
        # Add character info to the prompt so it stays consistent
        char_info = state.get("character_info", "")
        prompt = f"Character Info: {char_info}\n\n" + generate_prompt(state, player_action)
        prompt += (
            "\n\nNow give the player 3 choices for what they can do next. Format it like:\n"
            "1. [Choice 1]\n2. [Choice 2]\n3. [Choice 3]"
        )

        # Ask the model
        response = client.invoke(prompt)
        full_text = response.content

        # Separate story from choices (basic parsing)
        if "1." in full_text:
            story_part, options_block = full_text.split("1.", 1)
            option_lines = options_block.strip().split("\n")
            option_lines = ["1. " + option_lines[0]] + option_lines[1:3]  # limit to 3 options
        else:
            story_part = full_text
            option_lines = []

        # Store for rerendering in session
        st.session_state.story_part = story_part
        st.session_state.options = option_lines
        st.session_state.last_input = player_action
        st.rerun()

    # If we already have story + options from previous turn
    if "story_part" in st.session_state:
        st.markdown("### DM Continues:")
        st.write(st.session_state.story_part.strip())

    if "options" in st.session_state and st.session_state.options:
        with st.form("choice_form"):
            choice = st.radio("What do you do?", options=st.session_state.options)
            confirmed = st.form_submit_button("Confirm Choice")

        if confirmed:
            # Feed back the user's choice into the story
            followup = f"You chose: {choice}\nContinue the story from this choice."
            new_response = client.invoke(followup)
            dm_followup = new_response.content

            st.markdown("### DM Response:")
            st.write(dm_followup.strip())

            update_inventory(state, "placeholder item")  # You could add real logic here
            update_location(state, f"Branch: {choice[:30]}")
            add_to_history(state, st.session_state.last_input + f" ({choice})", dm_followup)

            # Clean up session to prepare for next turn
            del st.session_state.story_part
            del st.session_state.options
            del st.session_state.last_input
            st.rerun()

# Optional: Show story log so far
if state.get("history"):
    with st.expander("📜 Story So Far"):
        for entry in state["history"]:
            st.markdown(f"**You:** {entry['user']}")
            st.markdown(f"**DM:** {entry['dm']}\n")
