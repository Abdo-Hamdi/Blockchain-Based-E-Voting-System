import os
import warnings
import torch
import logging
import numpy as np
from PIL import Image
from typing import Optional
from facenet_pytorch import MTCNN, InceptionResnetV1

warnings.filterwarnings("ignore", category=FutureWarning, message=".*weights_only=False.*")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FaceAuthenticator:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        self.model = InceptionResnetV1(classify=False, pretrained='vggface2').to(self.device).eval()
        self.mtcnn = MTCNN(
            image_size=160,
            margin=14,
            device=self.device,
            selection_method='center_weighted_size',
            min_face_size=40)
        self._warmup()

    def _warmup(self):
        dummy = torch.randn(1, 3, 160, 160).to(self.device)
        with torch.no_grad():
            _ = self.model(dummy)

    def load_image(self, image_path: str) -> Optional[Image.Image]:
        try:
            if not os.path.exists(image_path):
                logger.error(f"Image not found: {image_path}")
                return None
                
            img = Image.open(image_path).convert('RGB')
            if min(img.size) < 40:
                logger.warning(f"Image too small: {img.size}")
                return None
            return img
        except Exception as e:
            logger.error(f"Error loading image: {str(e)}")
            return None

    def detect_face(self, image: Image.Image) -> Optional[torch.Tensor]:
        try:
            with torch.no_grad():
                face = self.mtcnn(image)
                if face is None:
                    logger.warning("No face detected")
                return face
        except Exception as e:
            logger.error(f"Face detection failed: {str(e)}")
            return None

    def create_embedding(self, image_path: str) -> Optional[np.ndarray]:
        try:
            img = self.load_image(image_path)
            if img is None:
                return None

            face_tensor = self.detect_face(img)
            if face_tensor is None:
                return None

            with torch.no_grad():
                embedding = self.model(face_tensor.unsqueeze(0).to(self.device))
                return embedding.cpu().numpy().flatten()
        except Exception as e:
            logger.error(f"Embedding generation failed: {str(e)}")
            return None

    @staticmethod
    def compare_embeddings(emb1: np.ndarray, emb2: np.ndarray) -> float:
        try:
            # Convert to numpy arrays if they aren't already
            emb1 = np.asarray(emb1).flatten()
            emb2 = np.asarray(emb2).flatten()
            
            # Safety checks
            if emb1.shape != emb2.shape:
                raise ValueError("Embedding dimension mismatch")
                
            # Compute cosine similarity
            dot_product = np.dot(emb1, emb2)
            norm_product = np.linalg.norm(emb1) * np.linalg.norm(emb2)
            
            # Handle division by zero
            if norm_product == 0:
                return 0.0
                
            similarity = dot_product / norm_product
            return float(np.clip(similarity, -1.0, 1.0))
            
        except Exception as e:
            logger.error(f"Comparison failed: {str(e)}")
            return -1.0  # Return invalid similarity on error

    def verify_user(self,similarity: float, threshold: float = 0.7) -> bool:
        return similarity >= threshold