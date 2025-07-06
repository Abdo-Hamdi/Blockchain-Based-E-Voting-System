#!/bin/bash

# Backend Startup Script
# Starts the face verification backend API server

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

print_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

# Main startup function
main() {
    echo "🚀 Starting Blockchain E-Voting Backend API..."
    echo "==============================================="
    
    # Check if we're in the correct directory
    if [ ! -f "Backend/app.py" ]; then
        print_error "Please run this script from the ai-models directory"
        print_info "Current directory: $(pwd)"
        print_info "Expected structure: ai-models/Backend/app.py"
        exit 1
    fi
    
    # Check Python version
    python_version=$(python3 --version 2>&1 | cut -d' ' -f2)
    major_version=$(echo $python_version | cut -d'.' -f1)
    minor_version=$(echo $python_version | cut -d'.' -f2)
    
    if [ "$major_version" -lt 3 ] || [ "$major_version" -eq 3 -a "$minor_version" -lt 8 ]; then
        print_error "Python 3.8+ is required. Found: $python_version"
        exit 1
    fi
    print_status "Python version: $python_version"
    
    # Check if virtual environment exists
    if [ ! -d "venv" ]; then
        print_warning "Virtual environment 'venv' not found!"
        print_info "Creating virtual environment..."
        python3 -m venv venv
        if [ $? -ne 0 ]; then
            print_error "Failed to create virtual environment"
            exit 1
        fi
        print_status "Virtual environment created"
    fi
    
    # Activate virtual environment
    print_info "Activating virtual environment..."
    source venv/bin/activate
    
    # Upgrade pip
    print_info "Upgrading pip..."
    pip install --upgrade pip > /dev/null 2>&1
    
    # Check if required packages are installed
    print_info "Checking dependencies..."
    python -c "import flask, cv2, torch, facenet_pytorch, easyocr, tensorflow" 2>/dev/null
    if [ $? -ne 0 ]; then
        print_warning "Missing dependencies! Installing requirements..."
        pip install -r Backend/requirements.txt
        if [ $? -ne 0 ]; then
            print_error "Failed to install dependencies"
            exit 1
        fi
        print_status "Dependencies installed successfully"
    else
        print_status "Dependencies verified"
    fi
    
    # Check GPU availability
    print_info "Checking GPU availability..."
    gpu_available=$(python -c "import torch; print(torch.cuda.is_available())" 2>/dev/null)
    if [ "$gpu_available" = "True" ]; then
        gpu_name=$(python -c "import torch; print(torch.cuda.get_device_name(0))" 2>/dev/null)
        print_status "GPU available: $gpu_name"
    else
        print_warning "GPU not available, using CPU"
    fi
    
    # Check model files
    print_info "Checking model files..."
    model_errors=0
    
    if [ ! -f "Models/ID Fake Detection/Fake_model_best.keras" ]; then
        print_error "Fake detection model not found: Models/ID Fake Detection/Fake_model_best.keras"
        model_errors=$((model_errors + 1))
    else
        print_status "Fake detection model found"
    fi
    
    if [ ! -f "Models/ID Detection & Field Extraction/detect_id_inside_image.pt" ]; then
        print_error "ID detection model not found: Models/ID Detection & Field Extraction/detect_id_inside_image.pt"
        model_errors=$((model_errors + 1))
    else
        print_status "ID detection model found"
    fi
    
    if [ ! -f "Models/ID Detection & Field Extraction/detect_field_inside_id.pt" ]; then
        print_error "Field detection model not found: Models/ID Detection & Field Extraction/detect_field_inside_id.pt"
        model_errors=$((model_errors + 1))
    else
        print_status "Field detection model found"
    fi
    
    if [ $model_errors -gt 0 ]; then
        print_error "Missing $model_errors model file(s). Please ensure all models are in place."
        print_info "See README.md for model setup instructions"
        exit 1
    fi
    
    # Test configuration
    print_info "Testing configuration..."
    cd Backend
    python -c "from config import validate_configuration; exit(0 if validate_configuration() else 1)"
    if [ $? -ne 0 ]; then
        print_error "Configuration validation failed"
        exit 1
    fi
    print_status "Configuration validated"
    
    # Start the server
    echo ""
    echo "🌐 Starting Flask API server..."
    echo "==============================================="
    echo "📋 Available endpoints:"
    echo "  - GET  /                    - API status and health check"
    echo "  - POST /extract_embeddings  - Extract face embeddings"
    echo "  - POST /verify_face         - Verify face against embeddings"
    echo "  - POST /verify_id_and_face  - Complete ID verification and face comparison"
    echo "  - POST /register            - Register new user with verification"
    echo "  - POST /login               - User login with face verification"
    echo ""
    echo "🔧 Configuration:"
    echo "  - Server: http://localhost:5000"
    echo "  - GPU: $([ "$gpu_available" = "True" ] && echo "Enabled" || echo "Disabled")"
    echo "  - Log Level: INFO"
    echo ""
    echo "Press Ctrl+C to stop the server"
    echo "==============================================="
    echo ""
    
    # Start the application
    python app.py
}

# Cleanup function
cleanup() {
    echo ""
    print_info "Shutting down server..."
    print_status "Server stopped successfully"
    exit 0
}

# Set up signal handlers
trap cleanup SIGINT SIGTERM

# Check if running as source or execution
if [ "${BASH_SOURCE[0]}" == "${0}" ]; then
    main "$@"
fi 