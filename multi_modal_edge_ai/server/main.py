from flask import Flask
from multi_modal_edge_ai.server.api.client_connection import client_connection_blueprint

# Initialize Flask application
app = Flask(__name__)

# Register blueprints
app.register_blueprint(client_connection_blueprint)

# you can use this instead of the terminal to run the server
if __name__ == '__main__':
    app.run()
