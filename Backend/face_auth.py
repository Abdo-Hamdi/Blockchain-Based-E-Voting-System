from config import *
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FaceAuthenticator:
    def __init__(self):
        self.device = DEVICE
        self.model = INCEPTION_MODEL
        self.mtcnn = mtcnn
        logger.info(f"Using device: {self.device}")

    def load_image(self, image_path: str) -> Optional[Image.Image]:
        try:
            if not os.path.exists(image_path):
                logger.error(f"Image not found: {image_path}")
                return None 
            img = Image.open(image_path).convert('RGB')
            if min(img.size) < MIN_IMAGE_SIZE:
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

    def create_embedding(self, image, path: bool = True) -> Optional[np.ndarray]:
        try:
            if path:
                img = self.load_image(image)
                if img is None:
                    return None
            else:
                img = image
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
    def compare_embeddings(emb1: np.ndarray, emb2: np.ndarray, threshold: float = SIMILARITY_THRESHOLD) -> Tuple[float, bool]:
        try:
            emb1 = np.asarray(emb1).flatten()
            emb2 = np.asarray(emb2).flatten()

            if emb1.shape != emb2.shape:
                raise ValueError("Embedding dimension mismatch")

            dot_product = np.dot(emb1, emb2)
            norm_product = np.linalg.norm(emb1) * np.linalg.norm(emb2)
            
            if norm_product == 0:
                return 0.0, False
                
            similarity = dot_product / norm_product
            return float(np.clip(similarity, -1.0, 1.0)), similarity >= threshold
            
        except Exception as e:
            logger.error(f"Comparison failed: {str(e)}")
            return -1.0, False