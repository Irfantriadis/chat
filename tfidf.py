from flask import Flask, render_template, request, jsonify
import json
import joblib

app = Flask(__name__)

# Load the JSON data
with open('model\\tegaltourism.json') as file:
    data = json.load(file)

# Load the trained model
model = joblib.load('model\chatbot_model.pkl')

def chatbot_response(input_text):
    predicted_tag = model.predict([input_text])[0]
    for intent in data['intents']:
        if intent['tag'] == predicted_tag:
            response = intent['responses']
            return response[0]
    return "Maaf, saya tidak mengerti apa yang Anda maksud."

@app.route('/')
def home():
    return render_template('chatbotidf.html')

@app.route('/get')
def get_bot_response():
    user_message = request.args.get('msg')
    return chatbot_response(user_message)

if __name__ == '__main__':
    app.run(debug=True)