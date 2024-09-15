import streamlit as st
import json
import joblib

# Load the JSON data
with open('model/tegaltourism.json') as file:
    data = json.load(file)

# Load the trained model
model = joblib.load('model/chatbot_model.pkl')

def chatbot_response(input_text):
    predicted_tag = model.predict([input_text])[0]
    for intent in data['intents']:
        if intent['tag'] == predicted_tag:
            response = intent['responses']
            return response[0]
    return "Maaf, saya tidak mengerti apa yang Anda maksud."

# Streamlit App Interface
st.title("Tegal Tourism Chatbot")
st.markdown("Ask questions about tourism spots and activities in Tegal!")

# Input field for the user question
user_input = st.text_input("Enter your question here:")

# Button to submit the question
if st.button("Submit"):
    if user_input:
        # Get chatbot response
        response = chatbot_response(user_input)
        # Display the response
        st.markdown(f"**Chatbot response:** {response}")
    else:
        st.warning("Please enter a question!")

