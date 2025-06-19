from config import *
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ImageExtractor:
    def __init__(self):
        self.device = DEVICE
        self.model1 = YOLO_MODEL_1
        self.mtcnn = mtcnn
        self.model2 = YOLO_MODEL_2
        logger.info(f"Using device: {self.device}")

    def load_image(self, image_path: str) -> Optional[Image.Image]:
        try:
            if not os.path.exists(image_path):
                logger.error(f"Image not found: {image_path}")
                return None 
            img = Image.open(image_path).convert('RGB')
            if min(img.size) < MIN_IMAGE_SIZE * 12:
                logger.warning(f"Image too small: {img.size}")
                return None
            return img
        except Exception as e:
            logger.error(f"Error loading image: {str(e)}")
            return None

    def detect_id(self, image: Image.Image) -> Optional[Image.Image]:
        try:
            with torch.no_grad():
                result = self.model1(np.array(image), verbose=False)
                boxes = result[0].boxes.xyxy.cpu().numpy() if len(result) > 0 else []
                if len(boxes) == 0:
                    logger.warning("No ID detected")
                    return None
                largest_box = max(boxes, key=lambda b: (b[2]-b[0]) * (b[3]-b[1]))
                x1, y1, x2, y2 = map(int, largest_box)
                id_crop = image.crop((x1, y1, x2, y2))
                return id_crop
        except Exception as e:
            logger.error(f"ID detection failed: {str(e)}")
            return None

    def extract_image(self, image_path: str) -> Optional[Image.Image]:
        try:
            img = self.load_image(image_path)
            if img is None:
                return None
            if min(img.size) < MIN_IMAGE_SIZE * 3:
                if img.width < 150:
                    img = img.resize((150, img.height), Image.BILINEAR)
                elif img.height < 100:
                    img = img.resize((img.width, 100), Image.BILINEAR)
                else:
                    img = img.resize((150, 100), Image.BILINEAR)
            id_crop = self.detect_id(img)
            if id_crop is None:
                return None        
            return id_crop
        except Exception as e:
            logger.error(f"Image extraction failed: {str(e)}")
            return None

    def extract_face(self, image: Image.Image) -> Optional[torch.Tensor]:
        try:
            with torch.no_grad():
                if image is None:
                    logger.error("No image provided for face extraction")
                    return None
                face_tensor = self.mtcnn(image)
                if face_tensor is None:
                    logger.warning("No face detected")
                    return None
                return face_tensor
        except Exception as e:
            logger.error(f"Face detection failed: {str(e)}")
            return None

    def extract_field(self, image: Image.Image) -> Optional[list]:
        try:
            with torch.no_grad():
                if image is None:
                    logger.error("No image provided for field extraction")
                    return None
                result = self.model2(np.array(image), verbose=False)
                boxes = result[0].boxes.xyxy.cpu().numpy() if len(result) > 0 else []
                if len(boxes) == 0:
                    logger.warning("No fields detected")
                    return None
                field_images = []
                for box in boxes:
                    x1, y1, x2, y2 = map(int, box)
                    field_crop = image.crop((x1, y1, x2, y2))
                    field_images.append(field_crop)
                return field_images
        except Exception as e:
            logger.error(f"Field detection failed: {str(e)}")
            return None