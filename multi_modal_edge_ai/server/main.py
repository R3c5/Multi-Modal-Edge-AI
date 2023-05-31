import logging
from logging.handlers import RotatingFileHandler

from flask import Flask
from multi_modal_edge_ai.server.api.client_connection import client_connection_blueprint

# Initialize Flask application
app = Flask(__name__)

# Configure logging
log_filename = 'app.log'
log_handler = RotatingFileHandler(log_filename, maxBytes=1000000, backupCount=1)
log_handler.setLevel(logging.INFO)
app.logger.addHandler(log_handler)

# Register blueprints
app.register_blueprint(client_connection_blueprint)

# you can use this instead of the terminal to run the server
if __name__ == '__main__':
    app.run()
