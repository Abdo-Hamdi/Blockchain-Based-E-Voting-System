from config import *
from face_auth import FaceAuthenticator

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
face_auth = FaceAuthenticator()
app.config.from_object('config')
# Swagger(app, config=app.config["SWAGGER"])

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config["ALLOWED_EXTENSIONS"]

@app.route('/')
def home():
    return jsonify({
        "api": "Face Authentication",
        "version": "1.0",
        "status": "running",
        "docs": "/apidocs",
        "try": "curl -X POST -F 'face_image=@selfie.jpg' http://localhost:5000/login"
    })

@app.route('/register', methods=['POST'])
def register():
    """Register a new user
    ---
    tags:
      - Authentication
    consumes:
      - multipart/form-data
    parameters:
      - name: face_image
        in: formData
        type: file
        required: true
        description: Face image for registration
      - name: user_address
        in: formData
        type: string
        required: true
        description: User's blockchain address
    responses:
      200:
        description: Registration successful
      400:
        description: Invalid input
    """
    return "Register"

@app.route('/login', methods=['POST'])
def login():
    """Authenticate user via face recognition
    ---
    tags:
      - Authentication
    consumes:
      - multipart/form-data
    parameters:
      - name: face_image
        in: formData
        type: file
        required: true
        description: Face image for login
      - name: user_address
        in: formData
        type: string
        required: true
        description: User's blockchain address
    responses:
      200:
        description: Authentication successful
        schema:
          type: object
          properties:
            access:
              type: string
              example: "granted"
            similarity:
              type: number
              example: 0.85
      401:
        description: Authentication failed
        schema:
          type: object
          properties:
            access:
              type: string
              example: "denied"
            similarity:
              type: number
              example: 0.45
            reason:
              type: string
              example: "Low similarity score"
      400:
        description: Bad request
      500:
        description: Internal server error
    """
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
        
        similarity = face_auth.compare_embeddings(stored_embedding, login_embedding)
        
        if similarity >= app.config["SIMILARITY_THRESHOLD"]:
            return jsonify({
                "access": "granted",
                "similarity": float(similarity)
            })
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