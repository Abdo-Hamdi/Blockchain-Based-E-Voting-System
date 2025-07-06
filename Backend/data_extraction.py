"""
Data Extraction Module

This module provides image processing capabilities for ID card detection,
fake detection, field extraction, and OCR for the blockchain-based e-voting system.

@title Data Extraction Service
@version 1.0.0
@description Core image processing and OCR services for ID verification
"""

from config import *
from typing import Optional, Tuple

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ImageExtractor:
    """
    @title Image Extraction Service
    @notice Handles ID detection, fake detection, field extraction, and OCR
    @dev Uses YOLO models for detection and TensorFlow for fake detection
    """
    
    def __init__(self):
        """
        @notice Initialize image extraction service
        @dev Sets up YOLO models and fake detection model
        """
        self.device = DEVICE
        self.model1 = YOLO_MODEL_1  # ID detection model
        self.model2 = YOLO_MODEL_2  # Field extraction model
        self.fake_model = FAKE_MODEL
        self.IMAGE_SIZE = 96
        logger.info(f"ImageExtractor initialized on device: {self.device}")

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
            if img is None or min(img.size) < MIN_IMAGE_SIZE:
                logger.warning(f"Image too small or invalid: {image_path}")
                return None
            
            logger.info(f"Image loaded successfully: {img.size}")
            return img
            
        except Exception as e:
            logger.error(f"Error loading image from {image_path}: {str(e)}")
            return None

    def detect_id(self, image: Image.Image) -> Optional[Image.Image]:
        """
        @notice Detect ID card in image and return cropped ID
        @param image PIL Image object containing ID card
        @return Cropped ID card image or None if not detected
        @dev Uses YOLO model to detect ID card boundaries
        """
        try:
            with torch.no_grad():
                result = self.model1(np.array(image), verbose=False)
                boxes = result[0].boxes.xyxy.cpu().numpy() if len(result) > 0 else []
                
                if len(boxes) == 0:
                    logger.warning("No ID detected in image")
                    return None
                
                # Select largest detected box (most likely the ID card)
                largest_box = max(boxes, key=lambda b: (b[2]-b[0]) * (b[3]-b[1]))
                x1, y1, x2, y2 = map(int, largest_box)
                id_crop = image.crop((x1, y1, x2, y2))
                
                logger.info(f"ID detected and cropped: {id_crop.size}")
                return id_crop
                
        except Exception as e:
            logger.error(f"ID detection failed: {str(e)}")
            return None

    def extract_id(self, image_path: str) -> Optional[Image.Image]:
        """
        @notice Extract ID card from image file
        @param image_path Path to image containing ID card
        @return Cropped ID card image or None if extraction fails
        @dev Handles image resizing if needed before detection
        """
        try:
            img = self.load_image(image_path)
            if img is None:
                return None
            
            # Resize image if too small for reliable detection
            min_width, min_height = 400, 200
            width, height = img.size
            
            if width < min_width or height < min_height:
                scale_w = min_width / width
                scale_h = min_height / height
                scale = max(scale_w, scale_h)
                new_width = max(int(width * scale), min_width)
                new_height = max(int(height * scale), min_height)
                img = img.resize((new_width, new_height), Image.BILINEAR)
                logger.info(f"Image resized from {(width, height)} to {img.size}")
            
            id_crop = self.detect_id(img)
            if id_crop is None:
                logger.warning("Failed to extract ID from image")
                return None
            
            logger.info("ID extracted successfully")
            return id_crop
            
        except Exception as e:
            logger.error(f"ID extraction failed: {str(e)}")
            return None

    def extract_field(self, image: Image.Image) -> Optional[Image.Image]:
        """
        @notice Extract National ID number field from ID card image
        @param image PIL Image object of ID card
        @return Cropped field image containing ID number or None if not found
        @dev Uses YOLO model to detect specific field within ID card
        """
        try:
            with torch.no_grad():
                if image is None:
                    logger.error("No image provided for field extraction")
                    return None
                
                result = self.model2(np.array(image), verbose=False)
                boxes = result[0].boxes
                
                # Look for ID number field (class ID_NUMBER_CLASS)
                for i in range(len(boxes)):
                    class_id = int(boxes.cls[i])
                    if class_id == ID_NUMBER_CLASS:
                        x1, y1, x2, y2 = map(int, boxes.xyxy[i])
                        field_crop = image.crop((x1, y1, x2, y2))
                        logger.info(f"ID number field extracted: {field_crop.size}")
                        return field_crop
                
                logger.warning("National ID number field not detected")
                return None
                
        except Exception as e:
            logger.error(f"Field extraction failed: {str(e)}")
            return None

    def load_balanced_image(self, image: Image.Image) -> Optional[np.ndarray]:
        """
        @notice Preprocess image for fake detection using balanced technique
        @param image PIL Image object
        @return Preprocessed image array or None if processing fails
        @dev Converts to grayscale, resizes, and normalizes for model input
        """
        try:
            img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
            if img is None:
                return None
            
            img = cv2.resize(img, (self.IMAGE_SIZE, self.IMAGE_SIZE))
            img = cv2.merge([img]*3)  # Convert to 3-channel
            img = img.astype(np.float32)
            img = (img / 127.5) - 1.0  # Normalize to [-1, 1]
            
            return img
            
        except Exception as e:
            logger.error(f"Balanced image processing failed: {str(e)}")
            return None

    def apply_balanced_ela(self, image: Image.Image) -> Optional[np.ndarray]:
        """
        @notice Apply Error Level Analysis (ELA) for fake detection
        @param image PIL Image object
        @return ELA-processed image array or None if processing fails
        @dev Compares original with JPEG-compressed version to detect manipulation
        """
        try:
            original = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
            if original is None:
                return None
            
            original = cv2.resize(original, (self.IMAGE_SIZE, self.IMAGE_SIZE))
            original_rgb = cv2.merge([original]*3)
            
            # Compress and decompress to create artifacts
            _, encoded = cv2.imencode('.jpg', original, [cv2.IMWRITE_JPEG_QUALITY, ELA_QUALITY])
            compressed = cv2.imdecode(encoded, cv2.IMREAD_GRAYSCALE)
            compressed = cv2.resize(compressed, (self.IMAGE_SIZE, self.IMAGE_SIZE))
            compressed_rgb = cv2.merge([compressed]*3)
            
            # Calculate difference and enhance
            diff = 255 - cv2.absdiff(original_rgb, compressed_rgb)
            ela_result = (diff * ELA_SCALE_FACTOR).astype(np.float32) / 255.0
            
            return ela_result
            
        except Exception as e:
            logger.error(f"ELA processing failed: {str(e)}")
            return None

    def apply_lbp(self, image: Image.Image) -> Optional[np.ndarray]:
        """
        @notice Apply Local Binary Pattern (LBP) for texture analysis
        @param image PIL Image object
        @return LBP-processed image array or None if processing fails
        @dev Extracts texture features using local binary patterns
        """
        try:
            gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
            if gray is None:
                return None
            
            gray = cv2.resize(gray, (self.IMAGE_SIZE, self.IMAGE_SIZE))
            
            # Apply LBP transformation
            lbp = local_binary_pattern(gray, P=LBP_POINTS, R=LBP_RADIUS, method='uniform')
            
            # Normalize and convert to RGB
            lbp_normalized = (lbp - lbp.min()) / (lbp.max() - lbp.min())
            lbp_rgb = cv2.merge([lbp_normalized.astype(np.float32)]*3)
            
            return lbp_rgb
            
        except Exception as e:
            logger.error(f"LBP processing failed: {str(e)}")
            return None

    def predict_fake(self, image: Image.Image, threshold: float = REAL_THRESHOLD) -> Tuple[float, bool]:
        """
        @notice Predict if ID card image is fake or real
        @param image PIL Image object of ID card
        @param threshold Threshold for real/fake classification
        @return Tuple of (confidence_score, is_real_boolean)
        @dev Uses ensemble of image processing techniques with ML model
        """
        try:
            if image is None:
                logger.error("No image provided for fake detection")
                return 0.0, False
                
            logger.info("🤖 Starting fake detection analysis...")
            logger.info(f"   🎯 Using threshold: {threshold}")
            logger.info(f"   📐 Input image size: {image.size}")
            
            # Process image with different techniques
            logger.info("🔄 Processing with balanced technique...")
            img1 = self.load_balanced_image(image)
            if img1 is None:
                logger.error("❌ Balanced image processing failed")
                return 0.0, False
            img1 = np.expand_dims(img1, axis=0)

            logger.info("🔄 Processing with ELA technique...")
            img2 = self.apply_balanced_ela(image)
            if img2 is None:
                logger.error("❌ ELA processing failed")
                return 0.0, False
            img2 = np.expand_dims(img2, axis=0)

            logger.info("🔄 Processing with LBP technique...")
            img3 = self.apply_lbp(image)
            if img3 is None:
                logger.error("❌ LBP processing failed")
                return 0.0, False
            img3 = np.expand_dims(img3, axis=0)

            # Run ML model prediction
            logger.info("🧠 Running ML model prediction...")
            prediction = self.fake_model.predict({
                'original_input': img1,
                'ela_input': img2,
                'lbp_input': img3
            }, verbose=0)[0][0]
                
            result = prediction > threshold
            
            # Log results
            logger.info(f"🎯 FAKE DETECTION RESULTS:")
            logger.info(f"   📊 Prediction score: {prediction:.6f}")
            logger.info(f"   📏 Threshold: {threshold}")
            logger.info(f"   🎪 Confidence: {prediction * 100:.2f}%")
            logger.info(f"   🔍 Classification: {'REAL' if result else 'FAKE'}")
            
            return float(prediction), result
            
        except Exception as e:
            logger.error(f"❌ Fake detection failed: {str(e)}")
            return 0.0, False

    def simple_threshold(self, image: np.ndarray) -> np.ndarray:
        """
        @notice Apply simple threshold for OCR preprocessing
        @param image Input image array
        @return Thresholded image array
        @dev Converts to grayscale and applies binary threshold
        """
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        _, thresh = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY_INV)
        return cv2.cvtColor(thresh, cv2.COLOR_GRAY2RGB)

    def convert_digits(self, text: str) -> str:
        """
        @notice Convert Arabic/Persian digits to English digits
        @param text Text containing digits in various scripts
        @return String with only English digits
        @dev Uses DIGIT_MAP for character conversion
        """
        return ''.join([DIGIT_MAP.get(c, '') for c in text if c in DIGIT_MAP])

    def run_ocr(self, image: Image.Image, text_threshold: float = 0.4) -> int:
        """
        @notice Extract National ID number from image using OCR
        @param image PIL Image object containing ID number
        @param text_threshold Confidence threshold for OCR text detection
        @return Extracted ID number as integer or 0 if extraction fails
        @dev Uses EasyOCR with Arabic and English support
        """
        if image is None:
            logger.error("No image provided for OCR")
            return 0
        
        try:
            # Prepare image for OCR
            image_np = np.array(image)
            if image_np.shape[-1] == 3:
                image_rgb = image_np
            else:
                image_rgb = cv2.cvtColor(image_np, cv2.COLOR_GRAY2RGB)
            
            # Preprocess image
            processed = self.simple_threshold(image_rgb)
            processed = cv2.resize(processed, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

            # Run OCR
            results = READER.readtext(
                processed,
                paragraph=True,
                batch_size=4,
                width_ths=1.5,
                text_threshold=text_threshold,
                allowlist=''.join(DIGIT_MAP.keys())
            )
            
            full_number = ""
            arabic_number = ""

            if results:
                # Select result with longest digit sequence
                best_result = max(results, key=lambda x: len(self.convert_digits(x[1])))
                arabic_number = best_result[1]
                full_number = self.convert_digits(arabic_number)
                
                # Validate ID number length
                if len(full_number) < 14:
                    logger.warning(f"Detected number too short ({len(full_number)} digits), likely invalid")
                    full_number = ""

            logger.info("=== OCR RESULTS ===")
            if full_number:
                logger.info(f"Raw Detection: {arabic_number}")
                logger.info(f"Converted Number: {full_number}")
                logger.info(f"Number Length: {len(full_number)}")
                return int(full_number)
            else:
                logger.warning("No valid ID number detected")
                return 0
                
        except Exception as e:
            logger.error(f"OCR processing failed: {str(e)}")
            return 0

    def process_complete_verification(self, image_path: str) -> Optional[dict]:
        """
        @notice Complete ID verification workflow
        @param image_path Path to image containing ID card
        @return Dictionary with verification results or None if failed
        @dev Combines all verification steps in one method
        """
        try:
            logger.info("🔄 Starting complete ID verification...")
            
            # Extract ID from image
            id_crop = self.extract_id(image_path)
            if id_crop is None:
                return None
            
            # Check if ID is fake
            prediction, is_real = self.predict_fake(id_crop)
            if not is_real:
                logger.error("❌ Fake ID detected")
                return {
                    "success": False,
                    "error": "Fake ID detected",
                    "fake_confidence": prediction
                }
            
            # Extract ID number field
            nid_field = self.extract_field(id_crop)
            if nid_field is None:
                return {
                    "success": False,
                    "error": "ID number field not found"
                }
            
            # Extract ID number using OCR
            id_number = self.run_ocr(nid_field)
            if id_number == 0:
                return {
                    "success": False,
                    "error": "Failed to extract ID number"
                }
            
            return {
                "success": True,
                "id_number": id_number,
                "authenticity_score": prediction,
                "id_crop": id_crop
            }
            
        except Exception as e:
            logger.error(f"Complete verification failed: {str(e)}")
            return None