import tensorflow as tf
import json
import random
import pickle
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()
intents = json.loads(open('model/tegaltourism.json').read())
words = pickle.load(open('model/words.pkl', 'rb'))
classes = pickle.load(open('model/classes.pkl', 'rb'))

ERROR_THRESHOLD = 0.5  # Definisikan ERROR_THRESHOLD di sini

def clean_up_sentence(sentence):
    if sentence is None:
        return []  # Mengembalikan daftar kosong jika kalimat adalah None
    # tokenize the pattern - split words into array
    sentence_words = nltk.word_tokenize(sentence)
    # stem each word - create short form for word
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

def bow(sentence, words, show_details=True):
    # tokenize the pattern
    sentence_words = clean_up_sentence(sentence)
    # bag of words - matrix of N words, vocabulary matrix
    bag = [0] * len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                # assign 1 if current word is in the vocabulary position
                bag[i] = 1
                if show_details:
                    print("found in bag: %s" % w)
    return np.array(bag)

def predict_class(sentence, interpreter):
    # tokenize the pattern
    sentence_words = clean_up_sentence(sentence)
    # bag of words
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

def getResponse(ints, intents_json):
    tag = ints[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if i['tag'] == tag:
            result = random.choice(i['responses'])
            break
    return result

def chatbot_response(msg):
    interpreter = tf.lite.Interpreter(model_path='model/chat_model.tflite')
    interpreter.allocate_tensors()
    global input_details, output_details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    ints = predict_class(msg, interpreter)
    if ints and float(ints[0]['probability']) >= ERROR_THRESHOLD:
        res = getResponse(ints, intents)
        response = res
    return response
