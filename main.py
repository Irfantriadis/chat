import streamlit as st
import tensorflow as tf
import json
import random
import pickle
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer

# Initialize the WordNetLemmatizer
lemmatizer = WordNetLemmatizer()

# Load intents, words, and classes from your pre-trained model
intents = json.loads(open('model/tegaltourism.json').read())
words = pickle.load(open('model/words.pkl', 'rb'))
classes = pickle.load(open('model/classes.pkl', 'rb'))

# Define the ERROR_THRESHOLD
ERROR_THRESHOLD = 0.5

# Function to clean and tokenize the user input
def clean_up_sentence(sentence):
    if sentence is None:
        return []
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

# Function to create a Bag of Words (BoW)
def bow(sentence, words, show_details=True):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1
                if show_details:
                    print("found in bag: %s" % w)
    return np.array(bag)

# Function to predict the intent class
def predict_class(sentence, interpreter):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1
    
    input_data = np.array([bag], dtype=np.float32)
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    
    results = [[i, r] for i, r in enumerate(output_data[0]) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list

# Function to get response from the bot
def getResponse(ints, intents_json):
    tag = ints[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if i['tag'] == tag:
            result = random.choice(i['responses'])
            break
    return result

# Function to generate chatbot response
def chatbot_response(msg):
    interpreter = tf.lite.Interpreter(model_path='model/chat_model.tflite')
    interpreter.allocate_tensors()
    global input_details, output_details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    ints = predict_class(msg, interpreter)
    if ints and float(ints[0]['probability']) >= ERROR_THRESHOLD:
        res = getResponse(ints, intents)
        return res
    return "I'm sorry, I didn't understand that."

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

