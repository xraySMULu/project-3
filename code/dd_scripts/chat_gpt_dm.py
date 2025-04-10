import os
import sys
from dotenv import load_dotenv
import openai

# Setup for relative path imports
sys.path.append(os.path.abspath("../code"))

from app.game_state import (
    load_state,
    update_inventory,
    update_location,
    add_to_history,
    generate_prompt
)

# Load your API key from the .env file
load_dotenv()
client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Load the current game state
state = load_state()

# Prompt the player for input
user_input = input("Player: ")

# Generate the prompt from the game state
prompt = generate_prompt(state, user_input)

# Get response from GPT-3.5 using OpenAI v1.x syntax
response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "system", "content": "You are a creative and vivid Dungeon Master."},
        {"role": "user", "content": prompt}
    ],
    temperature=0.9,
    max_tokens=300
)

# Extract the DM's reply from the response
dm_response = response.choices[0].message.content

# Print the DM response
print("\nDM:", dm_response)

# Update game state
update_inventory(state, "placeholder item")         # You can replace this with real logic later
update_location(state, "placeholder location")       # Or create a mapping system later
add_to_history(state, user_input, dm_response)