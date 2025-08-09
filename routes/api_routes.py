import logging
from flask import Blueprint, request, jsonify
from controller.predict_controller import PredictController

logger = logging.getLogger(__name__)
api_bp = Blueprint('api', __name__, url_prefix='/api')
controller = PredictController()

@api_bp.route('/predict', methods=['POST'])
def predict():
    try:
        logger.debug("Received prediction request")
        data = request.get_json()
        logger.debug(f"Request data: {data}")
        
        response, status_code = controller.predict_personality(data)
        logger.debug(f"Prediction result: {response}")
        
        return jsonify(response), status_code
    except Exception as e:
        logger.error(f"API error: {str(e)}", exc_info=True)
        return jsonify({"error": "Internal server error"}), 500
    
    