import os

class Config:
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'your-secret-key'
    MODEL_PATH = os.path.join(os.path.dirname(__file__), 'model', 'personality_model.pkl')
    PREPROCESSOR_PATH = os.path.join(os.path.dirname(__file__), 'model', 'preprocessor.pkl')