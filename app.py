from flask import Flask, render_template, request, jsonify
import chatbot  

app = Flask(__name__)
app.static_folder = 'static'

# @app.route("/")
# def index():
#     return render_template("index.html")

@app.route("/")
def chat():
    return render_template("chatbot2.html")

@app.route("/get", methods=["GET", "POST"])
def response():
    userText = request.args.get('msg')
    return chatbot.chatbot_response(userText)


if __name__ == "__main__":
    # app.run(debug=True)  
    app.run(host='0.0.0.0', port=5000, debug=True)
