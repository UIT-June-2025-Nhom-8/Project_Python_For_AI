#!/usr/bin/env python3
"""
Startup script for the AI model backend service
This script sets up the environment and starts the Flask server
"""

import sys
import os
import subprocess
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def check_python_version():
    """Check if Python version is compatible"""
    if sys.version_info < (3, 8):
        logger.error("Python 3.8 or higher is required")
        sys.exit(1)
    logger.info(f"Python version: {sys.version}")

def install_requirements():
    """Install required packages"""
    logger.info("Installing requirements...")
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "-r", "requirements.txt"
        ])
        logger.info("Requirements installed successfully")
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to install requirements: {e}")
        sys.exit(1)

def check_model_files():
    """Check if model files exist"""
    model_paths = [
        "../../models/sentiment_analyze_models/logistic_regression_20250820_173924.pkl",
        "../../models/gensim_lda_model/lda_model",
        "../../models/gensim_lda_model/dictionary.gensim"
    ]
    
    missing_files = []
    for path in model_paths:
        full_path = os.path.join(os.path.dirname(__file__), path)
        if not os.path.exists(full_path):
            missing_files.append(path)
    
    if missing_files:
        logger.error("Missing model files:")
        for file in missing_files:
            logger.error(f"  - {file}")
        logger.error("Please ensure all model files are present before starting the server")
        sys.exit(1)
    
    logger.info("All model files found")

def start_server():
    """Start the Flask server"""
    logger.info("Starting Flask server...")
    try:
        # Import and start the app
        from app import app
        app.run(debug=True, host='0.0.0.0', port=5000)
    except Exception as e:
        logger.error(f"Failed to start server: {e}")
        sys.exit(1)

if __name__ == "__main__":
    logger.info("=== AI Model Backend Service ===")
    
    # Check Python version
    check_python_version()
    
    # Install requirements
    install_requirements()
    
    # Check model files
    check_model_files()
    
    # Start server
    start_server()