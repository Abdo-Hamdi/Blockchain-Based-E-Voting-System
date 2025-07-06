#!/usr/bin/env python3
"""
Minimal test Flask app without heavy ML models
Use this to test if basic API functionality works
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import logging
import tempfile
import os

app = Flask(__name__)
CORS(app)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png'}
SIMILARITY_THRESHOLD = 0.7

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def home():
    return jsonify({
        "api": "Minimal Face Verification Test",
        "version": "1.0",
        "status": "running",
        "message": "This is a test version without ML models"
    })

@app.route('/verify_face', methods=['POST'])
def verify_face():
    """
    Test endpoint that simulates face verification without actual ML processing
    """
    logger.info("Received verify_face request")
    
    if 'face_image' not in request.files:
        logger.error("No image file in request")
        return jsonify({"error": "face_image is required"}), 400
    
    face_image = request.files['face_image']
    existing_embeddings_str = request.form.get('existing_embeddings')
    
    logger.info(f"Face image: {face_image.filename}")
    logger.info(f"Embeddings length: {len(existing_embeddings_str) if existing_embeddings_str else 0}")
    
    if not existing_embeddings_str:
        logger.error("No existing embeddings provided")
        return jsonify({"error": "existing_embeddings is required"}), 400
    
    if face_image.filename == '' or not allowed_file(face_image.filename):
        logger.error("Invalid file")
        return jsonify({"error": "Invalid or empty image file"}), 400
    
    temp_path = None        
    try:
        # Parse existing embeddings from JSON
        import json
        existing_embeddings = json.loads(existing_embeddings_str)
        logger.info(f"Parsed embeddings: {len(existing_embeddings)} dimensions")
        
        # Save uploaded image to temporary file (just to test file handling)
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as temp_file:
            temp_path = temp_file.name
            face_image.save(temp_path)
            logger.info(f"Saved image to: {temp_path}")

        # Simulate processing
        logger.info("Simulating face verification...")
        
        # Mock similarity score (0.8 for testing)
        similarity = 0.8
        is_match = similarity >= SIMILARITY_THRESHOLD
        
        # Log detailed verification results
        logger.info(f"Mock verification completed:")
        logger.info(f"  - Similarity score: {similarity:.4f}")
        logger.info(f"  - Threshold: {SIMILARITY_THRESHOLD}")
        logger.info(f"  - Match result: {is_match}")
        logger.info(f"  - Accuracy: {similarity * 100:.2f}%")
        
        return jsonify({
            "success": True,
            "is_match": is_match,
            "similarity": float(similarity),
            "threshold": SIMILARITY_THRESHOLD,
            "accuracy_percentage": round(similarity * 100, 2),
            "message": "Mock verification - Face verified successfully" if is_match else "Mock verification - Face verification failed"
        }), 200
                
    except json.JSONDecodeError:
        logger.error("JSON decode error")
        return jsonify({
            "success": False,
            "error": "Invalid existing_embeddings format. Must be a valid JSON array."
        }), 400
    except Exception as e:
        logger.error(f"Face verification error: {str(e)}", exc_info=True)
        return jsonify({
            "success": False,
            "error": "Face verification failed",
            "details": str(e)
        }), 500
        
    finally:
        # Clean up temporary file
        if temp_path and os.path.exists(temp_path):
            try:
                os.unlink(temp_path)
                logger.info("Cleaned up temp file")
            except Exception as e:
                logger.error(f"Temp file deletion failed: {str(e)}")

@app.route('/extract_embeddings', methods=['POST'])
def extract_embeddings():
    """
    Test endpoint that returns mock embeddings
    """
    logger.info("Received extract_embeddings request")
    
    if 'face_image' not in request.files:
        logger.error("No image file in request")
        return jsonify({"error": "face_image is required"}), 400
    
    face_image = request.files['face_image']
    
    if face_image.filename == '' or not allowed_file(face_image.filename):
        logger.error("Invalid file")
        return jsonify({"error": "Invalid or empty image file"}), 400
    
    # Return mock embeddings (512 dimensions like FaceNet)
    mock_embeddings = [0.1] * 512
    
    return jsonify({
        "success": True,
        "embeddings": mock_embeddings,
        "message": "Mock embeddings extracted successfully"
    }), 200

if __name__ == '__main__':
    print("🧪 Starting minimal test server...")
    print("This server uses mock data for testing API functionality")
    print("Access at: http://localhost:5000")
    app.run(host='0.0.0.0', port=5000, debug=True) 