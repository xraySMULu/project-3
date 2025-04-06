import openai
import json

openai.api.key = 'your api key'
fine_tuning_prompt = """[Conversation history]"""

with open('wizard_profiles.jsonl', 'r', encoding='utf-8') as infile:
    conversations = infile.readlines()
    for conversation in conversations:
        fine_tuning_prompt += conversation.strip() + "\n"

response = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": fine_tuning_prompt}
    ],
    temperature=0.7,
    max_tokens=1024,
    n=1,
    stop=None
)

model_id = response['id']
print(f"Model ID: {model_id}")
# Save the fine-tuned model
with open('fine_tuned_model.json', 'w') as outfile:
    json.dump(response, outfile)
# Load the fine-tuned model
with open('fine_tuned_model.json', 'r') as infile:
    fine_tuned_model = json.load(infile)
# Use the fine-tuned model for inference
response = openai.ChatCompletion.create(
    model=model_id,
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is the capital of France?"}
    ],
    temperature=0.7,
    max_tokens=1024,
    n=1,
    stop=None
)
print(response['choices'][0]['message']['content'])
# Save the inference result
with open('inference_result.json', 'w') as outfile:
    json.dump(response, outfile)

# Load the inference result
with open('inference_result.json', 'r') as infile:
    inference_result = json.load(infile)
# Print the inference result
print(inference_result['choices'][0]['message']['content'])
# Save the inference result to a file
with open('inference_result.txt', 'w') as outfile:
    outfile.write(inference_result['choices'][0]['message']['content'])

# Load the inference result from a file
with open('inference_result.txt', 'r') as infile:
    inference_result = infile.read()
# Print the inference result
print(inference_result)
# Save the inference result to a file
with open('inference_result.txt', 'w') as outfile:
    outfile.write(inference_result)




