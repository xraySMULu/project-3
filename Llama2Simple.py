import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

def run_chat_loop(model_name="meta-llama/Llama-2-7b-chat-hf"):
    """
    Loads the specified Llama 2 model in 4-bit quantization and starts an
    interactive console chat.
    """
    print(f"Loading model '{model_name}' in 4-bit quantization...")

    # 1. Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # 2. Configure 4-bit quantization
    quant_config = BitsAndBytesConfig(load_in_4bit=True)

    # 3. Load the model with 4-bit quantization
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quant_config,
        device_map="auto",       # Attempts to map layers automatically to GPU (if available)
        torch_dtype=torch.float16 # 16-bit floats help reduce memory usage
    )

    print("Model loaded! Type 'quit' or 'exit' to stop the chat loop.")

    # 4. Simple chat loop
    while True:
        user_input = input("\nUser: ")
        if user_input.lower() in ["quit", "exit"]:
            print("Exiting chat loop.")
            break

        # Tokenize user input and run inference
        inputs = tokenizer.encode(user_input, return_tensors="pt").to(model.device)
        with torch.no_grad():
            output_tokens = model.generate(
                inputs,
                max_new_tokens=128,
                do_sample=True,
                top_p=0.9,
                temperature=0.6
            )

        # Decode and print the response
        response_text = tokenizer.decode(output_tokens[0], skip_special_tokens=True)
        print(f"Model: {response_text}")

if __name__ == "__main__":
    run_chat_loop()