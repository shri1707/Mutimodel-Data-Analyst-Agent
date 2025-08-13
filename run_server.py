#!/usr/bin/env python3
"""
Production server runner for Data Analysis API.
Handles environment configuration and starts the FastAPI application with uvicorn.
"""

import os
import sys
import logging
from pathlib import Path

# Add the app directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'app'))

try:
    import uvicorn
    from dotenv import load_dotenv
except ImportError as e:
    print(f"Error importing required packages: {e}")
    print("Please ensure all dependencies are installed: pip install -r requirements.txt")
    sys.exit(1)

def setup_environment():
    """Load environment variables and set defaults for production."""
    # Load environment variables from .env file
    load_dotenv()
    
    # Set default values for production if not specified
    defaults = {
        'API_HOST': '0.0.0.0',
        'API_PORT': '8000',
        'API_WORKERS': '1',
        'LOG_LEVEL': 'INFO',
        'ENVIRONMENT': 'production',
        'DEBUG': 'False',
        'ENABLE_RELOAD': 'False'
    }
    
    for key, default_value in defaults.items():
        if key not in os.environ:
            os.environ[key] = default_value
    
    # Create required directories
    required_dirs = ['sandbox', 'logs', 'examples']
    for dir_name in required_dirs:
        Path(dir_name).mkdir(exist_ok=True)
    
    return {
        'host': os.getenv('API_HOST', '0.0.0.0'),
        'port': int(os.getenv('API_PORT', 8000)),
        'workers': int(os.getenv('API_WORKERS', 1)),
        'log_level': os.getenv('LOG_LEVEL', 'info').lower(),
        'reload': os.getenv('ENABLE_RELOAD', 'False').lower() == 'true',
        'debug': os.getenv('DEBUG', 'False').lower() == 'true'
    }

def check_required_env_vars():
    """Check that required environment variables are set."""
    required_vars = ['GEMINI_API_KEY']
    missing_vars = []
    
    for var in required_vars:
        if not os.getenv(var):
            missing_vars.append(var)
    
    if missing_vars:
        print(f"Error: Missing required environment variables: {', '.join(missing_vars)}")
        print("Please set these variables in your .env file or environment.")
        return False
    
    return True

def main():
    """Main entry point for the server."""
    print("🚀 Starting Data Analysis API Server...")
    
    # Setup environment and check requirements
    config = setup_environment()
    
    if not check_required_env_vars():
        sys.exit(1)
    
    # Configure logging
    logging.basicConfig(
        level=getattr(logging, config['log_level'].upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    print(f"📊 Environment: {os.getenv('ENVIRONMENT', 'production')}")
    print(f"🌐 Host: {config['host']}:{config['port']}")
    print(f"👥 Workers: {config['workers']}")
    print(f"📝 Log Level: {config['log_level']}")
    print(f"🔄 Reload: {config['reload']}")
    
    try:
        # Import the FastAPI app
        from app.main import app
        
        # Run the server
        uvicorn.run(
            "app.main:app",
            host=config['host'],
            port=config['port'],
            workers=config['workers'] if not config['reload'] else 1,  # Single worker for reload mode
            log_level=config['log_level'],
            reload=config['reload'],
            access_log=True,
            server_header=False,  # Security: don't reveal server info
            date_header=False     # Security: don't reveal date
        )
    
    except KeyboardInterrupt:
        print("\n🛑 Server stopped by user")
    except Exception as e:
        print(f"❌ Server failed to start: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
