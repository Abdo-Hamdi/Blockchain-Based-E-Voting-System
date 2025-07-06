# AI Models - Face Verification Backend

## Overview

This directory contains the AI-powered backend services for the blockchain-based e-voting system. The backend provides secure face verification, ID authentication, and fraud detection capabilities using advanced computer vision and machine learning techniques.

## 🔗 Related Repositories

- [Blockchain Smart Contract](https://github.com/suhail-abdelaal/Blockchain-based-eVoting)

## Features

- **Face Recognition**: Extract and compare facial embeddings using InceptionResnetV1
- **ID Card Detection**: Automatically detect and crop National ID cards using YOLO
- **Fake Detection**: Advanced fraud detection using ensemble ML techniques (ELA, LBP, CNN)
- **OCR Processing**: Extract National ID numbers with Arabic/English support
- **Real-time Processing**: Fast, GPU-accelerated inference
- **Secure API**: RESTful API with comprehensive error handling and logging

## Directory Structure

```
ai-models/
├── Backend/                    # Main backend application
│   ├── app.py                 # Flask API server
│   ├── config.py              # Configuration and model setup
│   ├── face_auth.py           # Face recognition module
│   ├── data_extraction.py     # ID processing and OCR
│   ├── requirements.txt       # Python dependencies
│   └── README.md             # Backend-specific documentation
├── Models/                    # Pre-trained models (required)
│   ├── ID Detection & Field Extraction/
│   │   ├── detect_id_inside_image.pt
│   │   └── detect_field_inside_id.pt
│   └── ID Fake Detection/
│       └── Fake_model_best.keras
├── Docs/                      # Documentation and presentations
├── start_backend.sh           # Automated startup script (Linux/macOS)
├── start_backend.bat          # Automated startup script (Windows)
└── README.md                 # This file
```

## Prerequisites

### System Requirements
- **Python**: 3.8 or higher
- **RAM**: Minimum 8GB (16GB recommended)
- **Storage**: At least 5GB free space for models and dependencies
- **GPU**: CUDA-compatible GPU (optional, for better performance)

### Operating System Support
- **Linux** (Ubuntu 18.04+, CentOS 7+) - Use `start_backend.sh`
- **macOS** (10.14+) - Use `start_backend.sh` 
- **Windows** (10/11) - Use `start_backend.bat` or Git Bash/WSL with `start_backend.sh`

## Quick Start

### 1. Clone and Navigate
```bash
cd ai-models
```

### 2. Run the Automated Setup

#### For Linux/macOS:
```bash
chmod +x start_backend.sh
./start_backend.sh
```

#### For Windows:
```cmd
start_backend.bat
```

#### Alternative for Windows (using Git Bash or WSL):
```bash
chmod +x start_backend.sh
./start_backend.sh
```

The script will automatically:
- Check Python version and system requirements
- Create a virtual environment
- Install all dependencies
- Validate model files
- Start the backend server

### 3. Verify Installation
Once started, the API will be available at `http://localhost:5000`

Test with:
```bash
curl http://localhost:5000
```

Or in Windows PowerShell:
```powershell
Invoke-RestMethod http://localhost:5000
```

## Manual Setup

If you prefer manual setup or encounter issues with the automated script:

### 1. Create Virtual Environment

#### Linux/macOS:
```bash
python3 -m venv venv
source venv/bin/activate
```

#### Windows:
```cmd
python -m venv venv
venv\Scripts\activate.bat
```

#### Windows PowerShell:
```powershell
python -m venv venv
venv\Scripts\Activate.ps1
```

### 2. Install Dependencies
```bash
pip install --upgrade pip
pip install -r Backend/requirements.txt
```

### 3. Verify Models
Ensure the following model files are present:
```
Models/
├── ID Detection & Field Extraction/
│   ├── detect_id_inside_image.pt      # ~50MB
│   └── detect_field_inside_id.pt      # ~50MB
└── ID Fake Detection/
    └── Fake_model_best.keras          # ~100MB
```

### 4. Start the Server

#### Linux/macOS:
```bash
cd Backend
python app.py
```

#### Windows:
```cmd
cd Backend
python app.py
```

## GPU Support

### CUDA Setup (Optional but Recommended)
For better performance, install CUDA support:

```bash
# Check if CUDA is available
python -c "import torch; print(torch.cuda.is_available())"

# Install CUDA-enabled PyTorch (if needed)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### Performance Comparison
- **CPU**: ~2-5 seconds per image
- **GPU**: ~0.5-1 second per image

## API Endpoints

### Health Check
```bash
GET /
```

### Face Embedding Extraction
```bash
POST /extract_embeddings
Content-Type: multipart/form-data

Parameters:
- face_image: Image file containing a face
```

### Face Verification
```bash
POST /verify_face
Content-Type: multipart/form-data

Parameters:
- face_image: Image file containing a face
- existing_embeddings: JSON array of previously extracted embeddings
```

### Complete ID and Face Verification
```bash
POST /verify_id_and_face
Content-Type: multipart/form-data

Parameters:
- id_image: Image file containing National ID card
- face_image: Image file containing user's face
```

### User Registration
```bash
POST /register
Content-Type: multipart/form-data

Parameters:
- face_image: User's face image
- nid_image: National ID card image
- id_number: User-provided ID number (14 digits)
- user_address: User's blockchain address
```

## Configuration

### Key Settings (Backend/config.py)
```python
# Face matching threshold
SIMILARITY_THRESHOLD = 0.85

# Fake detection threshold
REAL_THRESHOLD = 0.75

# Server configuration
HOST = '0.0.0.0'
PORT = 5000
```

### Environment Variables
```bash
# Optional: Set CUDA device
export CUDA_VISIBLE_DEVICES=0

# Optional: Set log level
export LOG_LEVEL=INFO
```

## Testing

### Test with Sample Images
```bash
# Test face embedding extraction
curl -X POST -F "face_image=@sample_face.jpg" http://localhost:5000/extract_embeddings

# Test ID verification
curl -X POST \
  -F "id_image=@sample_id.jpg" \
  -F "face_image=@sample_face.jpg" \
  http://localhost:5000/verify_id_and_face
```

### Expected Response Format
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

## Troubleshooting

### Common Issues

#### 1. Model Files Missing
```
Error: Model file not found
```
**Solution**: Ensure all model files are in the correct directories. Download from the project repository or contact the team.

#### 2. Memory Issues
```
Error: CUDA out of memory
```
**Solutions**:
- Reduce batch size in config
- Use CPU instead of GPU
- Close other applications

#### 3. Python Version
```
Error: Python 3.8+ required
```
**Solution**: Update Python or use pyenv/conda to manage versions.

#### 4. Permission Issues (Linux/macOS)
```
Error: Permission denied
```
**Solution**: 
```bash
chmod +x start_backend.sh
sudo chown -R $USER:$USER ai-models/
```

#### 5. Port Already in Use
```
Error: Address already in use
```
**Solutions**:
- Kill existing process: 
  - Linux/macOS: `sudo lsof -t -i:5000 | xargs kill`
  - Windows: `netstat -ano | findstr :5000` then `taskkill /PID <PID> /F`
- Change port in config.py
- Use different port: `python app.py --port 5001`

#### 6. Windows PowerShell Execution Policy
```
Error: Execution of scripts is disabled on this system
```
**Solution**:
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

#### 7. Windows Unicode Issues
```
Error: UnicodeDecodeError
```
**Solution**: Set environment variable:
```cmd
set PYTHONIOENCODING=utf-8
```

### Debug Mode
Enable debug logging:
```bash
export DEBUG=true
python Backend/app.py
```

### Check System Resources
```bash
# Check memory usage
free -h

# Check GPU status (if available)
nvidia-smi

# Check disk space
df -h
```

## Performance Optimization

### For Production Deployment
1. **Use GPU**: Enable CUDA for 3-5x faster processing
2. **Memory**: Allocate at least 16GB RAM
3. **Storage**: Use SSD for faster model loading
4. **Network**: Ensure stable internet for model downloads

### Scaling Considerations
- **Load Balancing**: Use nginx or similar for multiple instances
- **Caching**: Implement Redis for embedding storage
- **Database**: Consider PostgreSQL for user data
- **Monitoring**: Use Prometheus/Grafana for metrics

## Security Notes

1. **Input Validation**: All uploads are validated for type and size
2. **Temporary Files**: Automatically cleaned up after processing
3. **Logging**: Comprehensive audit trail without sensitive data
4. **Network**: Use HTTPS in production
5. **Authentication**: Implement API keys for production use

## Integration with Frontend

The backend is designed to work with the React frontend. Key integration points:

1. **CORS**: Enabled for cross-origin requests
2. **File Upload**: Supports multipart/form-data
3. **Error Handling**: Standardized error responses
4. **Progress**: Detailed logging for UI feedback

## Support

### Logs Location
- Application logs: Console output
- Error logs: Captured with full stack traces
- Access logs: Flask default logging

### Getting Help
1. Check logs for detailed error messages
2. Verify all prerequisites are met
3. Ensure model files are correctly placed
4. Test with sample images first

### Performance Monitoring
Monitor these metrics:
- Response times per endpoint
- Success/failure rates
- Memory and GPU usage
- Error frequencies by type

## License

This project is part of the blockchain-based e-voting system. See the main project license for terms and conditions.

---

**Quick Commands Reference:**

#### Linux/macOS:
```bash
# Start backend
./start_backend.sh

# Manual start
cd Backend && python app.py

# Check status
curl http://localhost:5000

# Stop server
Ctrl+C
```

#### Windows:
```cmd
REM Start backend
start_backend.bat

REM Manual start
cd Backend && python app.py

REM Check status (PowerShell)
Invoke-RestMethod http://localhost:5000

REM Stop server
Ctrl+C
``` 
