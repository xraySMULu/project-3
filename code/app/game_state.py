import os
os.environ["TRANSFORMERS_NO_TF"] = "1"  # force Transformers to ignore TensorFlow

import json
from transformers import pipeline

STATE_FILE = os.path.join(os.path.dirname(__file__), "state.json")
summarizer = pipeline("summarization", model="facebook/bart-large-cnn", framework="pt")

# Track how many turns have passed (stored in memory, not JSON)
turn_counter = 0
TURNS_BETWEEN_SUMMARIES = 3  # Change this to control how often we summarize

def load_state():
    with open(STATE_FILE, "r") as f:
        return json.load(f)

def save_state(state):
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)

def update_inventory(state, item, action="add"):
    if action == "add":
        state["inventory"].append(item)
        # Optional 1: Remove duplicates
        state["inventory"] = list(set(state["inventory"]))
    elif action == "remove" and item in state["inventory"]:
        state["inventory"].remove(item)
    save_state(state)

def update_location(state, new_location):
    state["location"] = new_location
    if new_location not in state["visited_locations"]:
        state["visited_locations"].append(new_location)
    save_state(state)

def add_to_history(state, user_input, dm_response):
    global turn_counter
    entry = {
        "user": user_input,
        "dm": dm_response
    }
    state["history"].append(entry)
    save_state(state)

    turn_counter += 1
    if turn_counter % TURNS_BETWEEN_SUMMARIES == 0:
        summarize_history(state)

def summarize_history(state, max_turns=10):
    """
    Summarizes the last X entries in the history and stores it in state["story_summary"]
    """
    recent_history = state["history"][-max_turns:]
    text_to_summarize = "\n".join(
        [f"Player: {entry['user']}\nDM: {entry['dm']}" for entry in recent_history]
    )
    if text_to_summarize.strip():
        summary_output = summarizer(text_to_summarize, max_length=60, min_length=20, do_sample=False)
        summary = summary_output[0]["summary_text"]
        state["story_summary"] = summary
        save_state(state)
        return summary
    else:
        return "Nothing to summarize yet."

def generate_prompt(state, user_input):
    memory_text = f"""
You are a Dungeon Master guiding a player through a fantasy world.

Current Location: {state.get('location', 'Unknown')}
Inventory: {', '.join(state.get('inventory', [])) or 'None'}
Visited Locations: {', '.join(state.get('visited_locations', [])) or 'None'}
Story Summary: {state.get('story_summary', 'None')}

Now continue the story based on the player's next action.

Player: {user_input}
DM:"""
    return memory_text.strip()