"""
Blockchain-Based E-Voting System - Face Verification API

This module provides REST API endpoints for face verification, ID authentication,
and user registration for a blockchain-based e-voting system.

@title Face Verification API
@version 1.0.0
@description API for face verification and ID authentication in blockchain e-voting
@contact support@evoting.system
"""

from config import *
from face_auth import FaceAuthenticator
from data_extraction import ImageExtractor
from flask_cors import CORS
from typing import Tuple, Optional, Dict, Any
from dataclasses import dataclass

# Initialize Flask app and configure CORS
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize core components
face_auth = FaceAuthenticator()
extractor = ImageExtractor()
app.config.from_object('config')

@dataclass
class ValidationResult:
    """Data class for validation results"""
    is_valid: bool
    error_message: Optional[str] = None
    details: Optional[Dict[str, Any]] = None

@dataclass
class ProcessingResult:
    """Data class for processing results"""
    success: bool
    data: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None

def allowed_file(filename: str) -> bool:
    """
    @notice Check if uploaded file has allowed extension
    @param filename The name of the file to check
    @return True if file extension is allowed, False otherwise
    """
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config["ALLOWED_EXTENSIONS"]

def validate_request_files(required_files: list) -> ValidationResult:
    """
    @notice Validate that required files are present in request
    @param required_files List of required file field names
    @return ValidationResult with validation status and error message if any
    """
    for file_key in required_files:
        if file_key not in request.files:
            return ValidationResult(
                is_valid=False,
                error_message=f"{file_key} is required"
            )
        
        file = request.files[file_key]
        if file.filename == '' or not allowed_file(file.filename):
            return ValidationResult(
                is_valid=False,
                error_message=f"Invalid or empty {file_key} file"
            )
    
    return ValidationResult(is_valid=True)

def validate_request_params(required_params: list) -> ValidationResult:
    """
    @notice Validate that required parameters are present in request
    @param required_params List of required parameter names
    @return ValidationResult with validation status and error message if any
    """
    for param in required_params:
        if not request.form.get(param):
            return ValidationResult(
                is_valid=False,
                error_message=f"{param} is required"
            )
    
    return ValidationResult(is_valid=True)

def save_temp_files(files: Dict[str, any]) -> Tuple[Dict[str, str], list]:
    """
    @notice Save uploaded files to temporary locations
    @param files Dictionary of file objects to save
    @return Tuple of (file_paths_dict, list_of_paths_for_cleanup)
    """
    temp_paths = {}
    cleanup_paths = []
    
    try:
        for key, file in files.items():
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')
            temp_path = temp_file.name
            temp_file.close()
            file.save(temp_path)
            temp_paths[key] = temp_path
            cleanup_paths.append(temp_path)
            logger.info(f"Saved {key} to temporary file: {temp_path}")
        
        return temp_paths, cleanup_paths
    except Exception as e:
        # Clean up any files that were successfully saved
        cleanup_temp_files(cleanup_paths)
        raise e

def cleanup_temp_files(file_paths: list) -> None:
    """
    @notice Clean up temporary files
    @param file_paths List of file paths to delete
    """
    for path in file_paths:
        if path and os.path.exists(path):
            try:
                os.unlink(path)
                logger.info(f"Cleaned up temporary file: {path}")
            except Exception as e:
                logger.error(f"Failed to delete temporary file {path}: {str(e)}")

def process_id_verification(id_image_path: str) -> ProcessingResult:
    """
    @notice Process ID verification including fake detection and field extraction
    @param id_image_path Path to the ID image file
    @return ProcessingResult with verification results or error
    """
    try:
        logger.info("🆔 Starting ID verification process...")
        
        # Step 1: Extract/crop ID from uploaded image
        logger.info("Step 1: Extracting ID from uploaded image...")
        id_crop = extractor.extract_id(id_image_path)
        if id_crop is None:
            return ProcessingResult(
                success=False,
                error_message="Could not detect ID card in the uploaded image. Please ensure the ID is clearly visible and try again."
            )
        
        # Step 2: Check if cropped ID is fake
        logger.info("Step 2: Checking if ID is authentic...")
        prediction, is_real = extractor.predict_fake(id_crop)
        
        if not is_real:
            logger.error(f"FAKE ID DETECTED - prediction: {prediction:.6f}")
            return ProcessingResult(
                success=False,
                error_message="Cannot process this ID - it appears to be fake",
                data={
                    "fake_confidence": float(prediction),
                    "threshold_used": REAL_THRESHOLD,
                    "confidence_percentage": round(prediction * 100, 2),
                    "processing_stopped_at": "fake_detection"
                }
            )
        
        # Step 3: Extract NID field from cropped ID
        logger.info("Step 3: Extracting NID field from ID...")
        nid_field = extractor.extract_field(id_crop)
        if nid_field is None:
            return ProcessingResult(
                success=False,
                error_message="Could not detect the National ID number field in the ID card. Please ensure the ID is clear and properly oriented."
            )
        
        # Step 4: Run OCR to extract NID number
        logger.info("Step 4: Running OCR to extract NID number...")
        extracted_nid = extractor.run_ocr(nid_field)
        if not extracted_nid or extracted_nid == 0:
            return ProcessingResult(
                success=False,
                error_message="Failed to extract National ID number from the card. Please ensure the ID number is clearly visible."
            )
        
        # Validate NID length
        nid_str = str(extracted_nid)
        if len(nid_str) != 14:
            return ProcessingResult(
                success=False,
                error_message=f"Extracted National ID number ({nid_str}) is not valid. Expected 14 digits, got {len(nid_str)}."
            )
        
        logger.info(f"✅ ID verification completed successfully: {nid_str}")
        return ProcessingResult(
            success=True,
            data={
                "extracted_nid": int(nid_str),
                "id_crop": id_crop,
                "authenticity_check": {
                    "is_real": is_real,
                    "confidence": float(prediction)
                }
            }
        )
        
    except Exception as e:
        logger.error(f"ID verification failed: {str(e)}", exc_info=True)
        return ProcessingResult(
            success=False,
            error_message="ID verification failed due to processing error"
        )

def process_face_comparison(face_image_path: str, id_crop: Image.Image) -> ProcessingResult:
    """
    @notice Compare face from live photo with face from ID card
    @param face_image_path Path to the live face image
    @param id_crop Cropped ID card image
    @return ProcessingResult with comparison results or error
    """
    try:
        logger.info("👤 Starting face comparison process...")
        
        # Extract face embeddings from live photo
        logger.info("Extracting face from live captured photo...")
        live_embedding = face_auth.create_embedding(face_image_path)
        if live_embedding is None:
            return ProcessingResult(
                success=False,
                error_message="Could not detect a face in the captured image. Please retake the photo ensuring your face is clearly visible."
            )
        
        # Extract face embeddings from ID card
        logger.info("Extracting face from ID card photo...")
        id_face_embedding = face_auth.create_embedding(id_crop, path=False)
        if id_face_embedding is None:
            return ProcessingResult(
                success=False,
                error_message="Could not detect a face in the ID card image. Please ensure the ID photo is clear and visible."
            )
        
        # Compare faces
        logger.info("Comparing faces...")
        similarity, faces_match = face_auth.compare_embeddings(live_embedding, id_face_embedding)
        
        logger.info(f"Face comparison results: similarity={similarity:.4f}, match={faces_match}")
        
        if not faces_match:
            return ProcessingResult(
                success=False,
                error_message="Face verification failed - the captured photo does not match the face in the ID card",
                data={
                    "similarity": float(similarity),
                    "threshold": SIMILARITY_THRESHOLD,
                    "accuracy_percentage": round(similarity * 100, 2)
                }
            )
        
        return ProcessingResult(
            success=True,
            data={
                "similarity": float(similarity),
                "threshold": SIMILARITY_THRESHOLD,
                "accuracy_percentage": round(similarity * 100, 2),
                "faces_match": faces_match
            }
        )
        
    except Exception as e:
        logger.error(f"Face comparison failed: {str(e)}", exc_info=True)
        return ProcessingResult(
            success=False,
            error_message="Face comparison failed due to processing error"
        )

def create_error_response(message: str, status_code: int = 400, details: Dict[str, Any] = None) -> Tuple[Dict[str, Any], int]:
    """
    @notice Create standardized error response
    @param message Error message
    @param status_code HTTP status code
    @param details Additional error details
    @return Tuple of (response_dict, status_code)
    """
    response = {
        "success": False,
        "error": message
    }
    if details:
        response.update(details)
    
    return response, status_code

def create_success_response(data: Dict[str, Any], message: str = None) -> Tuple[Dict[str, Any], int]:
    """
    @notice Create standardized success response
    @param data Response data
    @param message Success message
    @return Tuple of (response_dict, status_code)
    """
    response = {
        "success": True,
        **data
    }
    if message:
        response["message"] = message
    
    return response, 200

# API ENDPOINTS

@app.route('/')
def home():
    """
    @notice API information endpoint
    @return JSON response with API details and available endpoints
    """
    return jsonify({
        "api": "Blockchain Based E-Voting System - Face Verification",
        "version": "1.0",
        "status": "running",
        "endpoints": {
            "extract_embeddings": "POST /extract_embeddings - Extract face embeddings from image",
            "verify_face": "POST /verify_face - Compare face with existing embeddings",
            "verify_id_and_face": "POST /verify_id_and_face - Complete ID verification and face comparison",
            "register": "POST /register - Register user with ID and face verification"
        }
    })

@app.route('/extract_embeddings', methods=['POST'])
def extract_embeddings():
    """
    @notice Extract face embeddings from a single image
    @dev Workflow 1: Extract face embeddings from uploaded image
    @param face_image (file) Image file containing a face
    @return JSON response with embeddings array or error message
    """
    logger.info("🔍 EXTRACT_EMBEDDINGS endpoint called")
    
    # Validate request
    validation = validate_request_files(['face_image'])
    if not validation.is_valid:
        logger.error(f"Validation failed: {validation.error_message}")
        return jsonify(*create_error_response(validation.error_message))
    
    temp_paths = []
    try:
        # Save uploaded file
        files = {'face_image': request.files['face_image']}
        temp_file_paths, temp_paths = save_temp_files(files)
        
        # Extract embeddings
        embeddings = face_auth.create_embedding(temp_file_paths['face_image'])
        
        if embeddings is None:
            return jsonify(*create_error_response("No face detected in the image"))
        
        return jsonify(*create_success_response({
            "embeddings": embeddings.tolist(),
            "message": "Face embeddings extracted successfully"
        }))
        
    except Exception as e:
        logger.error(f"Embedding extraction error: {str(e)}", exc_info=True)
        return jsonify(*create_error_response(
            "Image processing failed", 
            500, 
            {"details": str(e)}
        ))
    finally:
        cleanup_temp_files(temp_paths)

@app.route('/verify_face', methods=['POST'])
def verify_face():
    """
    @notice Compare face image with existing embeddings
    @dev Workflow 2: Compare face image with stored embeddings
    @param face_image (file) Image file containing a face
    @param existing_embeddings (JSON array) Previously extracted embeddings
    @return JSON response with similarity score and verification result
    """
    logger.info("🔍 VERIFY_FACE endpoint called")
    
    # Validate request
    validation = validate_request_files(['face_image'])
    if not validation.is_valid:
        return jsonify(*create_error_response(validation.error_message))
    
    existing_embeddings_str = request.form.get('existing_embeddings')
    if not existing_embeddings_str:
        return jsonify(*create_error_response("existing_embeddings is required"))
    
    temp_paths = []
    try:
        # Parse existing embeddings
        logger.info("Parsing existing embeddings...")
        try:
            existing_embeddings_list = json.loads(existing_embeddings_str)
            existing_embeddings = np.array(existing_embeddings_list)
            logger.info(f"Embeddings parsed successfully: shape={existing_embeddings.shape}")
        except json.JSONDecodeError as e:
            return jsonify(*create_error_response(
                "Invalid existing_embeddings format. Must be a valid JSON array.",
                400,
                {"details": str(e)}
            ))
        
        # Save uploaded file
        files = {'face_image': request.files['face_image']}
        temp_file_paths, temp_paths = save_temp_files(files)
        
        # Extract embeddings from new image
        logger.info("Extracting embeddings from new image...")
        new_embeddings = face_auth.create_embedding(temp_file_paths['face_image'])
        
        if new_embeddings is None:
            return jsonify(*create_error_response("No face detected in the image"))
        
        # Compare embeddings
        logger.info("Comparing embeddings...")
        similarity, is_match = face_auth.compare_embeddings(existing_embeddings, new_embeddings)
        
        logger.info(f"Verification completed: similarity={similarity:.4f}, match={is_match}")
        
        return jsonify(*create_success_response({
            "is_match": is_match,
            "similarity": float(similarity),
            "threshold": SIMILARITY_THRESHOLD,
            "accuracy_percentage": round(similarity * 100, 2),
            "message": "Face verified successfully" if is_match else "Face verification failed"
        }))
        
    except Exception as e:
        logger.error(f"Face verification error: {str(e)}", exc_info=True)
        return jsonify(*create_error_response(
            "Face verification failed",
            500,
            {"details": str(e)}
        ))
    finally:
        cleanup_temp_files(temp_paths)

@app.route('/verify_id_and_face', methods=['POST'])
def verify_id_and_face():
    """
    @notice Complete ID verification and face comparison workflow
    @dev Performs ID detection, fake detection, OCR, and face comparison
    @param id_image (file) Image file containing National ID card
    @param face_image (file) Image file containing user's face
    @return JSON response with extracted NID and face comparison results
    """
    logger.info("🔍 VERIFY_ID_AND_FACE endpoint called")
    
    # Validate request
    validation = validate_request_files(['id_image', 'face_image'])
    if not validation.is_valid:
        return jsonify(*create_error_response(validation.error_message))
    
    temp_paths = []
    try:
        # Save uploaded files
        files = {
            'id_image': request.files['id_image'],
            'face_image': request.files['face_image']
        }
        temp_file_paths, temp_paths = save_temp_files(files)
        
        # Process ID verification
        id_result = process_id_verification(temp_file_paths['id_image'])
        if not id_result.success:
            return jsonify(*create_error_response(id_result.error_message, 400, id_result.data))
        
        # Process face comparison
        face_result = process_face_comparison(temp_file_paths['face_image'], id_result.data['id_crop'])
        if not face_result.success:
            return jsonify(*create_error_response(face_result.error_message, 400, face_result.data))
        
        # Success response
        logger.info("🎉 ID verification and face comparison completed successfully!")
        return jsonify(*create_success_response({
            "extracted_nid": id_result.data['extracted_nid'],
            "face_verification": face_result.data,
            "authenticity_check": id_result.data['authenticity_check'],
            "message": "ID verification and face comparison completed successfully"
        }))
        
    except Exception as e:
        logger.error(f"ID and face verification error: {str(e)}", exc_info=True)
        return jsonify(*create_error_response(
            "ID verification failed due to processing error",
            500,
            {"details": str(e)}
        ))
    finally:
        cleanup_temp_files(temp_paths)

@app.route('/register', methods=['POST'])
def register():
    """
    @notice Register user with comprehensive ID and face verification
    @dev Performs complete verification workflow and validates provided ID number
    @param face_image (file) Image file containing user's face
    @param nid_image (file) Image file containing National ID card
    @param id_number (string) User-provided ID number for validation
    @param user_address (string) User's blockchain address
    @return JSON response with registration result
    """
    logger.info("🔍 REGISTER endpoint called")
    
    # Validate request files and parameters
    file_validation = validate_request_files(['face_image', 'nid_image'])
    if not file_validation.is_valid:
        return jsonify(*create_error_response(file_validation.error_message))
    
    param_validation = validate_request_params(['id_number', 'user_address'])
    if not param_validation.is_valid:
        return jsonify(*create_error_response(param_validation.error_message))
    
    id_number = request.form.get('id_number')
    user_address = request.form.get('user_address')
    
    temp_paths = []
    try:
        # Save uploaded files
        files = {
            'face_image': request.files['face_image'],
            'nid_image': request.files['nid_image']
        }
        temp_file_paths, temp_paths = save_temp_files(files)
        
        # Process ID verification
        id_result = process_id_verification(temp_file_paths['nid_image'])
        if not id_result.success:
            return jsonify(*create_error_response(id_result.error_message, 400, id_result.data))
        
        # Validate provided ID number matches extracted ID
        extracted_nid = str(id_result.data['extracted_nid'])
        if extracted_nid != id_number:
            return jsonify(*create_error_response(
                "Provided ID number does not match the one extracted from the national ID image",
                400,
                {"provided_id": id_number, "extracted_id": extracted_nid}
            ))
        
        # Process face comparison
        face_result = process_face_comparison(temp_file_paths['face_image'], id_result.data['id_crop'])
        if not face_result.success:
            return jsonify(*create_error_response(face_result.error_message, 400, face_result.data))
        
        # Registration successful
        logger.info(f"✅ User registered successfully: {user_address} with ID {id_number}")
        return jsonify(*create_success_response({
            "user_address": user_address,
            "id_number": id_number,
            "face_verification": face_result.data,
            "authenticity_check": id_result.data['authenticity_check'],
            "message": "Registration completed successfully"
        }))
        
    except Exception as e:
        logger.error(f"Registration error: {str(e)}", exc_info=True)
        return jsonify(*create_error_response(
            "Registration failed",
            500,
            {"details": str(e)}
        ))
    finally:
        cleanup_temp_files(temp_paths)

@app.route('/login', methods=['POST'])
def login():
    """
    @notice Authenticate user using face verification
    @dev Currently incomplete - requires blockchain integration for stored embeddings
    @param face_image (file) Image file containing user's face
    @param user_address (string) User's blockchain address
    @return JSON response with authentication result
    @dev TODO: Implement blockchain integration to retrieve stored embeddings
    """
    logger.info("🔍 LOGIN endpoint called")
    
    # Validate request
    file_validation = validate_request_files(['face_image'])
    if not file_validation.is_valid:
        return jsonify(*create_error_response(file_validation.error_message))
    
    param_validation = validate_request_params(['user_address'])
    if not param_validation.is_valid:
        return jsonify(*create_error_response(param_validation.error_message))
    
    user_address = request.form.get('user_address')
    
    temp_paths = []
    try:
        # Save uploaded file
        files = {'face_image': request.files['face_image']}
        temp_file_paths, temp_paths = save_temp_files(files)
        
        # TODO: Retrieve stored embedding from blockchain
        # This is a placeholder - actual implementation needs blockchain integration
        stored_embedding = None
        
        if stored_embedding is None:
            return jsonify(*create_error_response(
                "User not found or blockchain integration not implemented",
                400,
                {"user_address": user_address}
            ))
        
        # Extract embeddings from login image
        login_embedding = face_auth.create_embedding(temp_file_paths['face_image'])
        if login_embedding is None:
            return jsonify({"access": "denied", "reason": "No face detected"}), 401
        
        # Compare embeddings
        similarity, is_valid = face_auth.compare_embeddings(stored_embedding, login_embedding)
        
        if is_valid:
            logger.info(f"✅ Login successful for user: {user_address}")
            return jsonify({
                "access": "granted",
                "similarity": float(similarity),
                "user_address": user_address
            }), 200
        else:
            logger.warning(f"❌ Login failed for user: {user_address}")
            return jsonify({
                "access": "denied",
                "similarity": float(similarity),
                "reason": "Low similarity score"
            }), 401
            
    except Exception as e:
        logger.error(f"Login error: {str(e)}", exc_info=True)
        return jsonify(*create_error_response(
            "Login failed",
            500,
            {"details": str(e)}
        ))
    finally:
        cleanup_temp_files(temp_paths)

if __name__ == '__main__':
    logger.info("🚀 Starting Face Verification API...")
    app.run(host=HOST, port=PORT, debug=DEBUG)