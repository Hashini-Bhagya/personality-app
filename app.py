from flask import Flask, render_template
from flask_cors import CORS  
from routes.api_routes import api_bp
import logging

app = Flask(__name__)
app.config.from_object('config.Config')


CORS(app)


app.register_blueprint(api_bp)


logging.basicConfig(level=logging.DEBUG)

@app.route('/')
def home():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)