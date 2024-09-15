import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Load model dan tokenizer
model_path = "D:\Chatbot TegalTourism\\tegal_tourism_chatbot"
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained(model_path)

def generate_response(prompt, max_length=100):
    full_prompt = f"Human: {prompt}\nBot:"
    input_ids = tokenizer.encode(full_prompt, return_tensors="pt")
    output = model.generate(input_ids, max_length=max_length, num_return_sequences=1, pad_token_id=tokenizer.eos_token_id)
    return tokenizer.decode(output[0], skip_special_tokens=True).split("Bot: ")[-1].strip()

# Contoh penggunaan
while True:
    user_input = input("You: ")
    if user_input.lower() == "quit":
        break
    response = generate_response(user_input)
    print("Bot:", response)