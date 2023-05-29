from flask import Flask, request

app = Flask(__name__)


# GET API
@app.route('/api/get_greeting', methods=['GET'])
def get_greeting():
    name = request.args.get('name')
    if name:
        return f"Hello, {name}!"
    else:
        return "No name provided."


# POST API
@app.route('/api/post_feedback', methods=['POST'])
def post_feedback():
    data = request.json
    if 'message' in data:
        name = data['name']
        message = data['message']
        print(f'Feedback for {name}: {message}')
        # Process the message or store it in a database
        return "Feedback received successfully!"
    else:
        return "Invalid feedback data."
