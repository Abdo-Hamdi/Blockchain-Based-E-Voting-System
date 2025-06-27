from config import *
from face_auth import FaceAuthenticator
from data_extraction import ImageExtractor

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
face_auth = FaceAuthenticator()
extractor = ImageExtractor()
app.config.from_object('config')
# Swagger(app, config=app.config["SWAGGER"])

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config["ALLOWED_EXTENSIONS"]

@app.route('/')
def home():
    return jsonify({
        "api": "Blockchain Based E-Voting System",
        "version": "1.0",
        "status": "running",
        "docs": "/apidocs",
        "try": "curl -X POST -F 'face_image=@selfie.jpg' http://localhost:5000/login"
    })

@app.route('/register', methods=['POST'])
def register():
    if 'face_image' not in request.files or 'nid_image' not in request.files:
        logger.error("Missing image file(s) in request")
        return jsonify({"error": "Both face_image and nid_image are required"}), 400

    face_image = request.files['face_image']
    nid_image = request.files['nid_image']
    id_number = request.form.get('id_number')
    user_address = request.form.get('user_address')

    if not user_address or not id_number:
        logger.error("Missing user_address or id_number")
        return jsonify({"error": "user_address and id_number are required"}), 400

    if face_image.filename == '' or not allowed_file(face_image.filename):
        logger.error("Invalid face image file")
        return jsonify({"error": "Invalid or empty face image file"}), 400

    if nid_image.filename == '' or not allowed_file(nid_image.filename):
        logger.error("Invalid national ID image file")
        return jsonify({"error": "Invalid or empty national ID image file"}), 400
    temp_face_path = None
    temp_nid_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as temp_file:
            temp_face_path = temp_file.name
            face_image.save(temp_face_path)
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as temp_file:
            temp_nid_path = temp_file.name
            nid_image.save(temp_nid_path)
        id_crop = extractor.extract_id(temp_nid_path)
        if id_crop is None:
            return jsonify({"error": "Could not detect ID card in national ID image"}), 400

        nid_field = extractor.extract_field(id_crop)
        if nid_field is None:
            return jsonify({"error": "Could not detect ID number field in national ID image"}), 400

        ocr_id_number = extractor.run_ocr(nid_field)
        if not ocr_id_number:
            return jsonify({"error": "OCR failed to extract ID number from national ID image"}), 400

        if ocr_id_number != id_number:
            return jsonify({
                "error": "Provided ID number does not match the one extracted from the national ID image",
                "ocr_id_number": ocr_id_number
            }), 400

        live_embedding = face_auth.create_embedding(temp_face_path)
        nid_embedding = face_auth.create_embedding(id_crop, path=False)

        if live_embedding is None or nid_embedding is None:
            return jsonify({"error": "Could not detect face in one or both images"}), 400
        similarity, is_valid = face_auth.compare_embeddings(live_embedding, nid_embedding)
        if not is_valid:
            return jsonify({
                "error": "Face images do not match",
                "similarity": float(similarity)
            }), 400
        # TODO: Save user_address, id_number, embeddings, etc. to your storage/blockchain
        # TODO: check if user_address already exists or id_number already exists
        logger.info(f"Registered user {user_address} with ID {id_number}")

        return jsonify({
            "message": "Registration successful",
            "user_address": user_address,
            "id_number": id_number
        }), 200

    except Exception as e:
        logger.error(f"Registration error: {str(e)}", exc_info=True)
        return jsonify({
            "error": "Registration failed",
            "details": str(e)
        }), 500

    finally:
        # Clean up temp files
        for path in [temp_face_path, temp_nid_path]:
            if path and os.path.exists(path):
                try:
                    os.unlink(path)
                except Exception as e:
                    logger.error(f"Temp file deletion failed: {str(e)}")

@app.route('/login', methods=['POST'])
def login():
    if 'face_image' not in request.files:
        logger.error("No image file in request")
        return jsonify({"error": "No image provided"}), 400
    
    face_image = request.files['face_image']
    user_address = request.form.get('user_address')
    
    if not user_address:
        logger.error("No user_address provided")
        return jsonify({"error": "user_address is required"}), 400
    
    if face_image.filename == '' or not allowed_file(face_image.filename):
        logger.error("Invalid file")
        return jsonify({"error": "Invalid or empty image file"}), 400
    
    temp_path = None        
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as temp_file:
            temp_path = temp_file.name
            face_image.save(temp_path)

        # TODO: Take the embedding from the blockchain
        stored_embedding = None
        
        login_embedding = face_auth.create_embedding(temp_path)
        if login_embedding is None:
            return jsonify({"access": "denied", "reason": "No face detected"}), 401
        
        similarity, is_valid = face_auth.compare_embeddings(stored_embedding, login_embedding)

        if is_valid:
            return jsonify({
                "access": "granted",
                "similarity": float(similarity)
            }), 200
        else:
            return jsonify({
                "access": "denied",
                "similarity": float(similarity),
                "reason": "Low similarity score"
            }), 401
                
    except Exception as e:
        logger.error(f"Processing error: {str(e)}", exc_info=True)
        return jsonify({
            "error": "Image processing failed",
            "details": str(e)
        }), 500
        
    finally:
        if temp_path and os.path.exists(temp_path):
            try:
                os.unlink(temp_path)
            except Exception as e:
                logger.error(f"Temp file deletion failed: {str(e)}")

if __name__ == '__main__':
    app.run(port=app.config["PORT"], debug=app.config["DEBUG"])