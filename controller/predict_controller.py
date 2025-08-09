import logging
from model.classifier import PersonalityClassifier
from utils.validation import validate_input

logger = logging.getLogger(__name__)

class PredictController:
    def __init__(self):
        try:
            logger.info("Initializing PersonalityClassifier")
            self.classifier = PersonalityClassifier()
            logger.info("Classifier initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize classifier: {str(e)}")
            raise
    
    def predict_personality(self, input_data):
        """Handle prediction request with detailed error handling"""
        try:
            
            if not validate_input(input_data):
                logger.warning("Invalid input data")
                return {"error": "Invalid input data"}, 400
            
            
            logger.debug("Making prediction")
            personality, confidence = self.classifier.predict(input_data)
            logger.info(f"Prediction: {personality} with {confidence*100:.1f}% confidence")
            
            return {
                "personality": personality,
                "confidence": confidence
            }, 200
        except Exception as e:
            logger.error(f"Prediction failed: {str(e)}", exc_info=True)
            return {"error": f"Prediction failed: {str(e)}"}, 500