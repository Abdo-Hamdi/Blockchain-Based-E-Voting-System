"""
Face Authentication Module

This module provides face detection, embedding extraction, and comparison functionality
for the blockchain-based e-voting system.

@title Face Authentication Service
@version 1.0.0
@description Core face authentication services for secure user verification
"""

from config import *
from typing import Optional, Tuple

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FaceAuthenticator:
    """
    @title Face Authentication Service
    @notice Handles face detection, embedding extraction, and comparison
    @dev Uses MTCNN for face detection and InceptionResnetV1 for embedding generation
    """
    
    def __init__(self):
        """
        @notice Initialize face authentication service
        @dev Sets up MTCNN detector and InceptionResnetV1 model
        """
        self.device = DEVICE
        self.model = INCEPTION_MODEL
        self.mtcnn = mtcnn
        logger.info(f"FaceAuthenticator initialized on device: {self.device}")

    def load_image(self, image_path: str) -> Optional[Image.Image]:
        """
        @notice Load and validate image from file path
        @param image_path Path to the image file
        @return PIL Image object or None if loading fails
        @dev Validates image exists and meets minimum size requirements
        """
        try:
            if not os.path.exists(image_path):
                logger.error(f"Image not found: {image_path}")
                return None 
            
            img = Image.open(image_path).convert('RGB')
            if min(img.size) < MIN_IMAGE_SIZE:
                logger.warning(f"Image too small: {img.size}, minimum: {MIN_IMAGE_SIZE}")
                return None
            
            logger.info(f"Image loaded successfully: {img.size}")
            return img
            
        except Exception as e:
            logger.error(f"Error loading image from {image_path}: {str(e)}")
            return None

    def detect_face(self, image: Image.Image) -> Optional[torch.Tensor]:
        """
        @notice Detect and extract face from image
        @param image PIL Image object containing a face
        @return Preprocessed face tensor or None if no face detected
        @dev Uses MTCNN for face detection and preprocessing
        """
        try:
            with torch.no_grad():
                face = self.mtcnn(image)
                if face is None:
                    logger.warning("No face detected in image")
                    return None
                
                logger.info(f"Face detected successfully: tensor shape {face.shape}")
                return face
                
        except Exception as e:
            logger.error(f"Face detection failed: {str(e)}")
            return None

    def create_embedding(self, image, path: bool = True) -> Optional[np.ndarray]:
        """
        @notice Generate face embedding from image
        @param image Either image path (if path=True) or PIL Image object (if path=False)
        @param path Boolean indicating if image parameter is a file path
        @return Face embedding as numpy array or None if processing fails
        @dev Uses InceptionResnetV1 model to generate 512-dimensional face embeddings
        """
        try:
            # Load image based on input type
            if path:
                img = self.load_image(image)
                if img is None:
                    return None
            else:
                img = image
                if img is None:
                    logger.error("No image provided for embedding generation")
                    return None

            # Detect face in image
            face_tensor = self.detect_face(img)
            if face_tensor is None:
                return None

            # Generate embedding
            with torch.no_grad():
                embedding = self.model(face_tensor.unsqueeze(0).to(self.device))
                embedding_np = embedding.cpu().numpy().flatten()
                
                logger.info(f"Embedding generated successfully: shape {embedding_np.shape}")
                return embedding_np
                
        except Exception as e:
            logger.error(f"Embedding generation failed: {str(e)}")
            return None

    @staticmethod
    def compare_embeddings(
        emb1: np.ndarray, 
        emb2: np.ndarray, 
        threshold: float = SIMILARITY_THRESHOLD
    ) -> Tuple[float, bool]:
        """
        @notice Compare two face embeddings for similarity
        @param emb1 First face embedding
        @param emb2 Second face embedding
        @param threshold Similarity threshold for match determination
        @return Tuple of (similarity_score, is_match_boolean)
        @dev Uses cosine similarity for comparison
        """
        try:
            logger.info("🔄 Starting embedding comparison...")
            logger.info(f"   - Embedding 1 type: {type(emb1)}, shape: {emb1.shape if hasattr(emb1, 'shape') else 'N/A'}")
            logger.info(f"   - Embedding 2 type: {type(emb2)}, shape: {emb2.shape if hasattr(emb2, 'shape') else 'N/A'}")
            
            # Ensure embeddings are numpy arrays and flattened
            emb1 = np.asarray(emb1).flatten()
            emb2 = np.asarray(emb2).flatten()
            
            logger.info(f"   - After flattening - emb1: {emb1.shape}, emb2: {emb2.shape}")
            logger.info(f"   - emb1 range: {emb1.min():.6f} to {emb1.max():.6f}")
            logger.info(f"   - emb2 range: {emb2.min():.6f} to {emb2.max():.6f}")

            # Validate embedding dimensions match
            if emb1.shape != emb2.shape:
                logger.error(f"❌ Embedding dimension mismatch: {emb1.shape} vs {emb2.shape}")
                raise ValueError("Embedding dimension mismatch")

            # Calculate cosine similarity
            logger.info("🔄 Computing cosine similarity...")
            dot_product = np.dot(emb1, emb2)
            norm_emb1 = np.linalg.norm(emb1)
            norm_emb2 = np.linalg.norm(emb2)
            norm_product = norm_emb1 * norm_emb2
            
            logger.info(f"   - Dot product: {dot_product:.6f}")
            logger.info(f"   - Norm emb1: {norm_emb1:.6f}")
            logger.info(f"   - Norm emb2: {norm_emb2:.6f}")
            logger.info(f"   - Norm product: {norm_product:.6f}")
            
            # Handle edge case of zero norm
            if norm_product == 0:
                logger.warning("⚠️ Zero norm product - returning 0 similarity")
                return 0.0, False
                
            # Calculate and validate similarity score
            similarity = dot_product / norm_product
            similarity_clipped = float(np.clip(similarity, -1.0, 1.0))
            is_match = similarity_clipped >= threshold
            
            logger.info(f"✅ Comparison completed:")
            logger.info(f"   - Raw similarity: {similarity:.6f}")
            logger.info(f"   - Clipped similarity: {similarity_clipped:.6f}")
            logger.info(f"   - Threshold: {threshold}")
            logger.info(f"   - Is match: {is_match}")
            
            return similarity_clipped, is_match
            
        except Exception as e:
            logger.error(f"❌ Embedding comparison failed: {str(e)}", exc_info=True)
            return -1.0, False

    def verify_face_from_paths(self, image_path1: str, image_path2: str) -> Tuple[float, bool]:
        """
        @notice Convenience method to verify faces from two image paths
        @param image_path1 Path to first image
        @param image_path2 Path to second image
        @return Tuple of (similarity_score, is_match_boolean)
        @dev Combines embedding extraction and comparison in one method
        """
        try:
            # Generate embeddings for both images
            emb1 = self.create_embedding(image_path1, path=True)
            emb2 = self.create_embedding(image_path2, path=True)
            
            if emb1 is None or emb2 is None:
                logger.error("Failed to generate embeddings from one or both images")
                return -1.0, False
            
            # Compare embeddings
            return self.compare_embeddings(emb1, emb2)
            
        except Exception as e:
            logger.error(f"Face verification from paths failed: {str(e)}")
            return -1.0, False