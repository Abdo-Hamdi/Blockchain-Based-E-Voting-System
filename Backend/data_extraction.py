from config import *
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ImageExtractor:
    def __init__(self):
        self.device = DEVICE
        self.model1 = YOLO_MODEL_1
        self.mtcnn = mtcnn
        self.model2 = YOLO_MODEL_2
        self.fake_model = FAKE_MODEL
        self.IMAGE_SIZE = 96
        logger.info(f"Using device: {self.device}")

    def load_image(self, image_path: str) -> Optional[Image.Image]:
        try:
            if not os.path.exists(image_path):
                logger.error(f"Image not found: {image_path}")
                return None 
            img = Image.open(image_path).convert('RGB')
            if img is None or min(img.size) < MIN_IMAGE_SIZE:
                logger.warning(f"Image not exists: {image_path}")
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

    def extract_id(self, image_path: str) -> Optional[Image.Image]:
        try:
            img = self.load_image(image_path)
            if img is None:
                return None
            if min(img.size) < MIN_IMAGE_SIZE * 3 and min(img.size) > MIN_IMAGE_SIZE:
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
                boxes = result[0].boxes
                for i in range(len(boxes)):
                    class_id = int(boxes.cls[i])
                    if class_id == ID_NUMBER_CLASS:
                        x1, y1, x2, y2 = map(int, boxes.xyxy[i])
                        field_crop = image.crop((x1, y1, x2, y2))
                        return field_crop
                logger.warning("National ID field not detected")
                return None
        except Exception as e:
            logger.error(f"Field detection failed: {str(e)}")
            return None

    def load_balanced_image(self, image: Image.Image) -> Optional[np.ndarray]:
        try:
            img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
            if img is None:
                return None
            img = cv2.resize(img, (self.IMAGE_SIZE, self.IMAGE_SIZE))
            img = cv2.merge([img]*3)
            img = img.astype(np.float32)
            img = (img / 127.5) - 1.0
            return img
        except Exception as e:
            logger.error(f"Error Processing image: {str(e)}")
            return None

    def apply_balanced_ela(self, image: Image.Image) -> Optional[np.ndarray]:
        try:
            original = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
            if original is None:
                return None
            original = cv2.resize(original, (self.IMAGE_SIZE, self.IMAGE_SIZE))
            original_rgb = cv2.merge([original]*3)
            _, encoded = cv2.imencode('.jpg', original, [cv2.IMWRITE_JPEG_QUALITY, ELA_QUALITY])
            compressed = cv2.imdecode(encoded, cv2.IMREAD_GRAYSCALE)
            compressed = cv2.resize(compressed, (self.IMAGE_SIZE, self.IMAGE_SIZE))
            compressed_rgb = cv2.merge([compressed]*3)
            diff = 255 - cv2.absdiff(original_rgb, compressed_rgb)
            return (diff * ELA_SCALE_FACTOR).astype(np.float32) / 255.0
        except Exception as e:
            logger.error(f"ELA failed: {str(e)}")
            return None

    def apply_lbp(self, image: Image.Image) -> Optional[np.ndarray]:
        try:
            gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
            if gray is None:
                return None
            gray = cv2.resize(gray, (self.IMAGE_SIZE, self.IMAGE_SIZE))
            lbp = local_binary_pattern(gray, P=LBP_POINTS, R=LBP_RADIUS, method='uniform')
            lbp_normalized = (lbp - lbp.min()) / (lbp.max() - lbp.min())
            lbp_rgb = cv2.merge([lbp_normalized.astype(np.float32)]*3)
            return lbp_rgb
        except Exception as e:
            logger.error(f"LBP failed: {str(e)}")
            return None

    def predict_fake(self, image: Image.Image, threshold: float = REAL_THRESHOLD):
        try:
            if image is None:
                logger.error("No image provided")
                return 0.0, False, 0.0
            img1 = self.load_balanced_image(image)
            if img1 is None:
                return 0.0, False, 0.0
            img1 = np.expand_dims(img1, axis=0)

            img2 = self.apply_balanced_ela(image)
            if img2 is None:
                return 0.0, False, 0.0
            img2 = np.expand_dims(img2, axis=0)

            img3 = self.apply_lbp(image)
            if img3 is None:
                return 0.0, False, 0.0
            img3 = np.expand_dims(img3, axis=0)

            prediction = self.fake_model.predict({
                'original_input': img1,
                'ela_input': img2,
                'lbp_input': img3
                })[0][0]
            result = True if prediction > threshold else False
            logger.info(f"Prediction: {prediction}, Result: {'Real' if result else 'Fake'}")
            confidence = max(prediction, 1 - prediction) * 100
            return float(prediction), result, confidence
        except Exception as e:
            logger.error(f"Prediction failed: {str(e)}")
            return 0.0, False, 0.0
    
    def simple_threshold(self, image: Image.Image) -> Optional[np.ndarray]:
        if image is None:
            logger.error("No image provided for OCR")
            return None
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        _, thresh = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY_INV)
        threshold = cv2.cvtColor(thresh, cv2.COLOR_GRAY2RGB)
        return threshold

    def convert_digits(self, text) -> str:
        return ''.join([DIGIT_MAP.get(c, '') for c in text if c in DIGIT_MAP])

    def run_ocr(self, image: Image.Image) -> str:
        if image is None:
            logger.error("No image provided for OCR")
            return ""
        try:
            image_np = np.array(image)
            # If image is already RGB, skip conversion, else convert as needed
            if image_np.shape[-1] == 3:
                image_rgb = image_np
            else:
                image_rgb = cv2.cvtColor(image_np, cv2.COLOR_GRAY2RGB)
            processed = self.simple_threshold(image_rgb)
            processed = cv2.resize(processed, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

            results = READER.readtext(
                processed,
                paragraph=True,
                batch_size=4,
                width_ths=1.5,
                text_threshold=0.4,
                allowlist=''.join(DIGIT_MAP.keys())
            )
            full_number = ""
            arabic_number = ""

            if results:
                best_result = max(results, key=lambda x: len(self.convert_digits(x[1])))
                arabic_number = best_result[1]
                full_number = self.convert_digits(arabic_number)
                if len(full_number) < 14:
                    logger.warning("Detected number is too short, likely not a valid ID number")
                    full_number = "" 

            logger.info("\n=== OCR RESULTS ===")
            if full_number:
                logger.info(f"Raw Detection: {arabic_number}")
                logger.info(f"Converted Number: {full_number}")
            else:
                logger.warning("No valid number detected")
            return full_number
        except Exception as e:
            logger.error(f"OCR failed: {str(e)}")
            return ""