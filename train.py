import random
import json
from fuzzywuzzy import process

# Dataset chatbot
with open('model/tegaltourism.json', 'r') as file:
    intents = json.load(file)

def get_intent_response(user_input):
    user_input = user_input.lower()
    
    best_match = None
    highest_score = 0
    
    # Cek setiap intent
    for intent in intents['intents']:
        tag = intent['tag']
        patterns = intent['patterns']
        responses = intent['responses']
        
        # Mencocokkan input pengguna dengan pola menggunakan fuzzy matching
        for pattern in patterns:
            score = process.extractOne(user_input, [pattern])[1]
            if score > highest_score:
                highest_score = score
                best_match = random.choice(responses)
    
    if highest_score > 60:  # Ambang batas kemiripan
        return best_match
    else:
        return "Maaf, saya tidak mengerti pertanyaan Anda."

# Loop chatbot
def chat():
    print("Halo! Saya chatbot. Ketik 'exit' untuk keluar.")
    while True:
        user_input = input("Anda: ")
        if user_input.lower() == 'exit':
            print("Chatbot: Sampai jumpa!")
            break
        response = get_intent_response(user_input)
        print(f"Chatbot: {response}")

if __name__ == "__main__":
    chat()
