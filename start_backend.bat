@echo off
REM Backend Startup Script for Windows
REM Starts the face verification backend API server

echo 🚀 Starting Blockchain E-Voting Backend API...
echo ===============================================

REM Check if we're in the correct directory
if not exist "Backend\app.py" (
    echo ❌ Please run this script from the ai-models directory
    echo Current directory: %CD%
    echo Expected structure: ai-models\Backend\app.py
    pause
    exit /b 1
)

REM Check Python version
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python is not installed or not in PATH
    echo Please install Python 3.8+ from https://python.org
    pause
    exit /b 1
)

REM Get Python version
for /f "tokens=2" %%i in ('python --version 2^>^&1') do set PYTHON_VERSION=%%i
echo ✅ Python version: %PYTHON_VERSION%

REM Check if virtual environment exists
if not exist "venv\" (
    echo ⚠️  Virtual environment 'venv' not found!
    echo ℹ️  Creating virtual environment...
    python -m venv venv
    if errorlevel 1 (
        echo ❌ Failed to create virtual environment
        pause
        exit /b 1
    )
    echo ✅ Virtual environment created
)

REM Activate virtual environment
echo ℹ️  Activating virtual environment...
call venv\Scripts\activate.bat

REM Upgrade pip
echo ℹ️  Upgrading pip...
python -m pip install --upgrade pip >nul 2>&1

REM Check if required packages are installed
echo ℹ️  Checking dependencies...
python -c "import flask, cv2, torch, facenet_pytorch, easyocr, tensorflow" >nul 2>&1
if errorlevel 1 (
    echo ⚠️  Missing dependencies! Installing requirements...
    pip install -r Backend\requirements.txt
    if errorlevel 1 (
        echo ❌ Failed to install dependencies
        pause
        exit /b 1
    )
    echo ✅ Dependencies installed successfully
) else (
    echo ✅ Dependencies verified
)

REM Check GPU availability
echo ℹ️  Checking GPU availability...
for /f %%i in ('python -c "import torch; print(torch.cuda.is_available())" 2^>nul') do set GPU_AVAILABLE=%%i
if "%GPU_AVAILABLE%"=="True" (
    for /f "tokens=*" %%i in ('python -c "import torch; print(torch.cuda.get_device_name(0))" 2^>nul') do set GPU_NAME=%%i
    echo ✅ GPU available: !GPU_NAME!
) else (
    echo ⚠️  GPU not available, using CPU
)

REM Check model files
echo ℹ️  Checking model files...
set MODEL_ERRORS=0

if not exist "Models\ID Fake Detection\Fake_model_best.keras" (
    echo ❌ Fake detection model not found: Models\ID Fake Detection\Fake_model_best.keras
    set /a MODEL_ERRORS+=1
) else (
    echo ✅ Fake detection model found
)

if not exist "Models\ID Detection & Field Extraction\detect_id_inside_image.pt" (
    echo ❌ ID detection model not found: Models\ID Detection ^& Field Extraction\detect_id_inside_image.pt
    set /a MODEL_ERRORS+=1
) else (
    echo ✅ ID detection model found
)

if not exist "Models\ID Detection & Field Extraction\detect_field_inside_id.pt" (
    echo ❌ Field detection model not found: Models\ID Detection ^& Field Extraction\detect_field_inside_id.pt
    set /a MODEL_ERRORS+=1
) else (
    echo ✅ Field detection model found
)

if %MODEL_ERRORS% gtr 0 (
    echo ❌ Missing %MODEL_ERRORS% model file(s). Please ensure all models are in place.
    echo ℹ️  See README.md for model setup instructions
    pause
    exit /b 1
)

REM Test configuration
echo ℹ️  Testing configuration...
cd Backend
python -c "from config import validate_configuration; exit(0 if validate_configuration() else 1)" >nul 2>&1
if errorlevel 1 (
    echo ❌ Configuration validation failed
    pause
    exit /b 1
)
echo ✅ Configuration validated

REM Start the server
echo.
echo 🌐 Starting Flask API server...
echo ===============================================
echo 📋 Available endpoints:
echo   - GET  /                    - API status and health check
echo   - POST /extract_embeddings  - Extract face embeddings
echo   - POST /verify_face         - Verify face against embeddings
echo   - POST /verify_id_and_face  - Complete ID verification and face comparison
echo   - POST /register            - Register new user with verification
echo   - POST /login               - User login with face verification
echo.
echo 🔧 Configuration:
echo   - Server: http://localhost:5000
if "%GPU_AVAILABLE%"=="True" (
    echo   - GPU: Enabled
) else (
    echo   - GPU: Disabled
)
echo   - Log Level: INFO
echo.
echo Press Ctrl+C to stop the server
echo ===============================================
echo.

REM Start the application
python app.py

REM Cleanup on exit
echo.
echo ℹ️  Shutting down server...
echo ✅ Server stopped successfully
pause 