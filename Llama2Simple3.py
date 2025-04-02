import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

# 1. Define your system prompt/role instructions
SYSTEM_PROMPT = (
    "You are a grand fantasy game master AI. "
    "You narrate an adventure story in direct response to the user's latest input. "
    "Pay close attention to the theme and tone of the user's initial input and persistently follow them."
    "Start your side of the story without greetings or introductions and absolutely no words such as 'great', instead jumping right into the action."
    "At the end of your response, please give the user 3 choices to choose from to continue the story."
    "Please make sure that each of the three options is only one sentence long and is distinctly different from the other two choices, and"
    "make sure the choices do not expound too heavily on contents of the story that would not otherwise be revealed in other choices."
    "Allow the user to dictate the flow of the story; do not provide too much unprompted and unambiguous information or context."
    "Keep track of the user's choices and maintain consistency, but DO NOT repeat the system prompt "
    "or restate the entire conversation. Please limit each response to about four sentences (and "
    "roughly 200 tokens) so that answers are concise and don't trail off mid-sentence."
)

def build_prompt(system_prompt, conversation_history, user_input):
    """
    Construct a prompt that includes:
      - System instructions
      - Previous user and AI messages
      - The latest user message
    """
    # Build the 'history' string from all previous turns
    # We'll keep them simple: "User: ...\nAI: ..."
    history_str = ""
    for (old_user_input, old_ai_output) in conversation_history:
        history_str += f"User: {old_user_input}\nAI: {old_ai_output}\n"

    # Llama-2 Chat style:
    # <s>[INST] <<SYS>> SYSTEM_PROMPT <</SYS>> HISTORY + user's new input [/INST]
    # For simplicity, we keep everything in one [INST] block.
    prompt = (
        f"<s>[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n"
        f"{history_str}"
        f"User: {user_input}\n[/INST]"
    )
    return prompt

def strip_leading_repeats(full_response):
    """
    Attempt to remove any repeated system/user text from the model's final output.
    1. Find the last occurrence of '[/INST]' if it exists, and only keep text after that.
    2. Remove repeated 'User:' or 'AI:' lines, if the model still includes them.
    """
    # 1) If the model echoed the entire prompt, it might appear before or after the last [/INST].
    if '[/INST]' in full_response:
        # Keep only what's after the final [/INST]
        full_response = full_response.split('[/INST]')[-1]

    # 2) Remove repeated references to system or user lines.
    #    For example, if the text has 'User:' or 'AI:' repeated at the start.
    #    We'll do a basic cleanup by stripping typical tokens.
    remove_terms = ["User:", "AI:", "<<SYS>>", "<</SYS>>", "[INST]"]
    for term in remove_terms:
        if term in full_response:
            # Repeatedly remove them if they appear at the start
            while full_response.strip().startswith(term):
                full_response = full_response.strip()
                full_response = full_response[len(term):].strip()

    return full_response.strip()

def run_chat_loop(model_name="meta-llama/Llama-2-7b-chat-hf"):
    print(f"Loading model '{model_name}' in 4-bit quantization...")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    quant_config = BitsAndBytesConfig(load_in_4bit=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quant_config,
        device_map="auto",
        torch_dtype=torch.float16
    )

    print("Model loaded! Type 'quit' or 'exit' to stop the chat loop.\n")

    conversation_history = []

    while True:
        user_input = input("User: ")
        if user_input.lower() in ["quit", "exit"]:
            print("Exiting chat loop.")
            break

        # Build the prompt with system text, past conversation, & new user input
        prompt = build_prompt(SYSTEM_PROMPT, conversation_history, user_input)

        # Encode prompt and generate
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            output_tokens = model.generate(
                input_ids,
                max_new_tokens=220,        # Hard cap on new tokens
                do_sample=True,
                top_p=0.9,
                temperature=0.6,
                # Optionally you could add e.g. repetition_penalty=1.02, etc.
            )

        # Decode the model's output
        full_response = tokenizer.decode(output_tokens[0], skip_special_tokens=True)

        # Clean up the output so it doesn't echo system/user lines
        response_text = strip_leading_repeats(full_response)

        # Print the final, cleaned AI response
        print(f"AI: {response_text}\n")

        # Update conversation memory
        conversation_history.append((user_input, response_text))

if __name__ == "__main__":
    run_chat_loop()