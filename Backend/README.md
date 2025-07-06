# Blockchain E-Voting System - Backend API

## Overview

This is the backend API for the blockchain-based e-voting system, providing secure face verification and ID authentication services. The system uses advanced computer vision and machine learning techniques to ensure voter identity verification and prevent fraud.

## Features

- **Face Detection & Verification**: Extract and compare facial embeddings using InceptionResnetV1
- **ID Card Authentication**: Detect and verify National ID cards using YOLO models
- **Fake Detection**: Identify fraudulent IDs using ensemble ML techniques (ELA, LBP, CNN)
- **OCR Processing**: Extract National ID numbers with Arabic/English support
- **Comprehensive Logging**: Detailed logging for monitoring and debugging
- **Secure File Handling**: Automatic cleanup of temporary files
- **CORS Support**: Cross-origin resource sharing for web applications

## Architecture

```
Backend/
├── app.py              # Main Flask application with API endpoints
├── config.py           # Configuration settings and model initialization
├── face_auth.py        # Face detection and verification module
├── data_extraction.py  # ID processing and OCR module
├── requirements.txt    # Python dependencies
└── README.md          # This documentation
```

## Models Required

The system requires the following pre-trained models:

1. **ID Detection Model**: `detect_id_inside_image.pt` (YOLO)
2. **Field Extraction Model**: `detect_field_inside_id.pt` (YOLO)
3. **Fake Detection Model**: `Fake_model_best.keras` (TensorFlow)
4. **Face Recognition Model**: InceptionResnetV1 (automatically downloaded)

## Installation

### Prerequisites

- Python 3.8+
- CUDA-compatible GPU (optional, for better performance)
- Required model files in the `Models/` directory

### Setup

1. **Clone the repository and navigate to the backend directory:**
   ```bash
   cd ai-models/Backend
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Verify model files are in place:**
   ```
   ../Models/
   ├── ID Detection & Field Extraction/
   │   ├── detect_id_inside_image.pt
   │   └── detect_field_inside_id.pt
   └── ID Fake Detection/
       └── Fake_model_best.keras
   ```

4. **Start the server:**
   ```bash
   python app.py
   ```

The API will be available at `http://localhost:5000`

## API Endpoints

### 1. Health Check
```
GET /
```
Returns API status and available endpoints.

**Response:**
```json
{
  "api": "Blockchain Based E-Voting System - Face Verification",
  "version": "1.0",
  "status": "running",
  "endpoints": {
    "extract_embeddings": "POST /extract_embeddings",
    "verify_face": "POST /verify_face",
    "verify_id_and_face": "POST /verify_id_and_face",
    "register": "POST /register"
  }
}
```

### 2. Extract Face Embeddings
```
POST /extract_embeddings
```
Extract facial embeddings from an image.

**Parameters:**
- `face_image` (file): Image containing a face

**Response:**
```json
{
  "success": true,
  "embeddings": [0.123, -0.456, ...],
  "message": "Face embeddings extracted successfully"
}
```

### 3. Verify Face
```
POST /verify_face
```
Compare a face image with existing embeddings.

**Parameters:**
- `face_image` (file): Image containing a face
- `existing_embeddings` (JSON string): Previously extracted embeddings

**Response:**
```json
{
  "success": true,
  "is_match": true,
  "similarity": 0.92,
  "threshold": 0.85,
  "accuracy_percentage": 92.0,
  "message": "Face verified successfully"
}
```

### 4. Complete ID and Face Verification
```
POST /verify_id_and_face
```
Perform complete verification: ID detection, fake detection, field extraction, OCR, and face comparison.

**Parameters:**
- `id_image` (file): Image containing National ID card
- `face_image` (file): Image containing user's face

**Response:**
```json
{
  "success": true,
  "extracted_nid": 12345678901234,
  "face_verification": {
    "similarity": 0.91,
    "threshold": 0.85,
    "accuracy_percentage": 91.0,
    "faces_match": true
  },
  "authenticity_check": {
    "is_real": true,
    "confidence": 0.89
  },
  "message": "ID verification and face comparison completed successfully"
}
```

### 5. User Registration
```
POST /register
```
Register a new user with complete verification.

**Parameters:**
- `face_image` (file): User's face image
- `nid_image` (file): National ID card image
- `id_number` (string): User-provided ID number for validation
- `user_address` (string): User's blockchain address

**Response:**
```json
{
  "success": true,
  "user_address": "0x123...",
  "id_number": "12345678901234",
  "face_verification": {
    "similarity": 0.88,
    "threshold": 0.85,
    "accuracy_percentage": 88.0,
    "faces_match": true
  },
  "authenticity_check": {
    "is_real": true,
    "confidence": 0.91
  },
  "message": "Registration completed successfully"
}
```

## Error Handling

All endpoints return standardized error responses:

```json
{
  "success": false,
  "error": "Error description",
  "details": "Additional error details (optional)"
}
```

Common error codes:
- `400`: Invalid request parameters or fake ID detected
- `500`: Internal server error during processing

## Configuration

### Key Configuration Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `SIMILARITY_THRESHOLD` | 0.85 | Face matching threshold |
| `REAL_THRESHOLD` | 0.75 | Fake detection threshold |
| `MIN_IMAGE_SIZE` | 100 | Minimum image size for processing |
| `HOST` | '0.0.0.0' | Server host |
| `PORT` | 5000 | Server port |

### Model Configuration

Models are automatically loaded on startup. The system validates:
- Model file existence
- Configuration parameter ranges
- Device availability (CUDA/CPU)

## Security Features

1. **Input Validation**: All file uploads are validated for type and size
2. **Temporary File Cleanup**: Automatic cleanup of uploaded files
3. **Fake Detection**: Multi-technique fake ID detection (ELA, LBP, CNN)
4. **Comprehensive Logging**: Detailed audit trail of all operations
5. **Error Handling**: Graceful error handling with informative messages

## Performance Considerations

- **GPU Acceleration**: Automatically uses CUDA if available
- **Batch Processing**: Optimized for single-image processing
- **Memory Management**: Efficient cleanup of temporary files
- **Model Caching**: Models loaded once at startup

## Logging

The system provides comprehensive logging:

- **INFO**: Normal operation, successful processing
- **WARNING**: Non-critical issues, fallback behaviors
- **ERROR**: Processing failures, validation errors

Logs include:
- Request processing steps
- Model prediction results
- Performance metrics
- Error details and stack traces

## Development

### Code Structure

The codebase follows clean architecture principles:

- **Separation of Concerns**: Each module has a specific responsibility
- **Error Handling**: Consistent error handling across all endpoints
- **Documentation**: Comprehensive NatSpec-style documentation
- **Type Hints**: Full type annotations for better IDE support

### Helper Functions

Key helper functions include:
- `validate_request_files()`: Input validation
- `save_temp_files()`: Secure file handling
- `process_id_verification()`: ID processing workflow
- `process_face_comparison()`: Face verification workflow

## Testing

### Manual Testing

Use tools like curl or Postman to test endpoints:

```bash
# Test health check
curl http://localhost:5000/

# Test face embedding extraction
curl -X POST -F "face_image=@test_face.jpg" http://localhost:5000/extract_embeddings
```

### Error Scenarios

The system handles various error conditions:
- Invalid file formats
- Missing faces in images
- Fake ID detection
- OCR failures
- Network issues

## Monitoring

Monitor the following metrics:
- Processing success rates
- Response times
- Model accuracy
- Error frequencies
- Resource usage

## Future Enhancements

Planned improvements include:
- Batch processing support
- Advanced anti-spoofing techniques
- Real-time processing optimization
- Enhanced fake detection models
- Blockchain integration for user storage

## Support

For issues or questions:
1. Check the logs for detailed error information
2. Verify model files are properly installed
3. Ensure all dependencies are correctly installed
4. Check GPU drivers if using CUDA

## License

This project is part of the blockchain-based e-voting system. See the main project license for terms and conditions. 