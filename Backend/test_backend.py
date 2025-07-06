#!/usr/bin/env python3
"""
Simple test script to diagnose backend issues
"""

import sys
import traceback

def test_imports():
    """Test if all required packages can be imported"""
    print("Testing imports...")
    
    try:
        import torch
        print(f"✅ PyTorch: {torch.__version__}")
        print(f"   CUDA available: {torch.cuda.is_available()}")
        print(f"   Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    except Exception as e:
        print(f"❌ PyTorch import failed: {e}")
        return False
    
    try:
        import cv2
        print(f"✅ OpenCV: {cv2.__version__}")
    except Exception as e:
        print(f"❌ OpenCV import failed: {e}")
        return False
    
    try:
        import facenet_pytorch
        print(f"✅ FaceNet PyTorch available")
    except Exception as e:
        print(f"❌ FaceNet PyTorch import failed: {e}")
        return False
    
    try:
        import flask
        print(f"✅ Flask: {flask.__version__}")
    except Exception as e:
        print(f"❌ Flask import failed: {e}")
        return False
    
    try:
        import tensorflow as tf
        print(f"✅ TensorFlow: {tf.__version__}")
    except Exception as e:
        print(f"❌ TensorFlow import failed: {e}")
        return False
    
    return True

def test_config():
    """Test if config.py can be imported and models loaded"""
    print("\nTesting config...")
    
    try:
        import config
        print("✅ Config imported successfully")
        
        # Test key variables
        print(f"   Device: {config.DEVICE}")
        print(f"   Similarity threshold: {config.SIMILARITY_THRESHOLD}")
        print(f"   Image size: {config.IMAGE_SIZE}")
        
        return True
    except Exception as e:
        print(f"❌ Config import failed: {e}")
        traceback.print_exc()
        return False

def test_face_auth():
    """Test if FaceAuthenticator can be initialized"""
    print("\nTesting FaceAuthenticator...")
    
    try:
        from face_auth import FaceAuthenticator
        auth = FaceAuthenticator()
        print("✅ FaceAuthenticator initialized successfully")
        return True
    except Exception as e:
        print(f"❌ FaceAuthenticator initialization failed: {e}")
        traceback.print_exc()
        return False

def test_data_extraction():
    """Test if ImageExtractor can be initialized"""
    print("\nTesting ImageExtractor...")
    
    try:
        from data_extraction import ImageExtractor
        extractor = ImageExtractor()
        print("✅ ImageExtractor initialized successfully")
        return True
    except Exception as e:
        print(f"❌ ImageExtractor initialization failed: {e}")
        traceback.print_exc()
        return False

def test_flask_app():
    """Test if Flask app can be created"""
    print("\nTesting Flask app...")
    
    try:
        from flask import Flask
        app = Flask(__name__)
        print("✅ Flask app created successfully")
        return True
    except Exception as e:
        print(f"❌ Flask app creation failed: {e}")
        traceback.print_exc()
        return False

def main():
    print("🔍 Backend Diagnostic Test")
    print("=" * 50)
    
    all_tests_passed = True
    
    # Run all tests
    tests = [
        test_imports,
        test_config,
        test_face_auth,
        test_data_extraction,
        test_flask_app
    ]
    
    for test in tests:
        if not test():
            all_tests_passed = False
            print()
    
    print("\n" + "=" * 50)
    if all_tests_passed:
        print("🎉 All tests passed! Backend should work properly.")
        print("Try starting the server with: python app.py")
    else:
        print("❌ Some tests failed. Please fix the issues above.")
        print("Make sure you have activated the virtual environment:")
        print("  source ../fv-env/bin/activate")
    
    return 0 if all_tests_passed else 1

if __name__ == "__main__":
    sys.exit(main()) 