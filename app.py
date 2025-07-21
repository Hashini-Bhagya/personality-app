from flask import Flask, render_template
from flask_cors import CORS  # Add this import
from routes.api_routes import api_bp
import logging

app = Flask(__name__)
app.config.from_object('config.Config')

# Enable CORS for all routes
CORS(app)

# Register blueprints
app.register_blueprint(api_bp)

# Configure logging
logging.basicConfig(level=logging.DEBUG)

@app.route('/')
def home():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)