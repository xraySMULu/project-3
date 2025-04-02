import sys
import os

# Add /code to the path so we can import from app/
sys.path.append(os.path.abspath("../code"))

from app.game_state import (
    load_state,
    update_inventory,
    update_location,
    add_to_history,
    generate_prompt,
    summarize_history  # ← new summarizer function
)

# Step 1: Load the current state
state = load_state()

# Step 2: Simulate a player action
user_input = "I open the ancient door with the torch raised."
dm_response = "The door creaks open slowly, revealing a dusty library filled with shadows."

# Step 3: Update the game state
update_inventory(state, "mysterious book", "add")
update_location(state, "Abandoned Library")
add_to_history(state, user_input, dm_response)

# Step 4: Summarize the story so far (last 10 turns)
summary = summarize_history(state)
print("\n--- STORY SUMMARY ---")
print(summary)
print("--- END SUMMARY ---\n")

# Step 5: Generate a full prompt with the current state
prompt = generate_prompt(state, user_input)
print("\n--- GENERATED PROMPT ---")
print(prompt)
print("--- END PROMPT ---\n")