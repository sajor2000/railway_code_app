#!/usr/bin/env python3
"""Basic server test to verify FastAPI startup without heavy dependencies."""

import os
import sys
from pathlib import Path

# Add backend to Python path
backend_path = Path(__file__).parent / "backend"
sys.path.insert(0, str(backend_path))

# Set minimal environment variables for testing
os.environ["ENVIRONMENT"] = "production"
os.environ["DEBUG"] = "false"
os.environ["PORT"] = "8080"

try:
    from fastapi import FastAPI
    from backend.config import get_settings
    
    # Test basic config loading
    settings = get_settings()
    print(f"✅ Config loaded: {settings.app_name} v{settings.app_version}")
    print(f"✅ Port: {settings.api_port}")
    print(f"✅ Environment: {settings.environment}")
    
    # Create minimal app
    test_app = FastAPI(title="Test Medical API", version="1.0.0")
    
    @test_app.get("/health")
    def health_check():
        return {"status": "healthy", "port": os.getenv("PORT", "8080")}
    
    @test_app.get("/api/health")
    def api_health_check():
        return {"status": "healthy", "service": "medical-coding-api"}
    
    print("✅ Basic FastAPI app created successfully")
    
    if __name__ == "__main__":
        import uvicorn
        print(f"🚀 Starting test server on port {settings.api_port}")
        uvicorn.run(test_app, host="0.0.0.0", port=settings.api_port)
        
except Exception as e:
    print(f"❌ Server test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)