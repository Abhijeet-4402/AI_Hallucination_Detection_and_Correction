"""
Setup script for AI Hallucination Detection System
Member 2: LLM & Detection Module

This script helps set up the environment and install dependencies.
"""

import os
import sys
import subprocess
from pathlib import Path

def create_directories():
    """Create necessary directories"""
    directories = [
        "src/member2_detection",
        "src/member1_retrieval", 
        "src/member3_correction",
        "src/member4_frontend",
        "src/shared",
        "tests",
        "data",
        "logs"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✅ Created directory: {directory}")

def create_env_file():
    """Create .env file if it doesn't exist"""
    env_file = Path(".env")
    env_example = Path(".env.example")
    
    if not env_file.exists() and env_example.exists():
        # Copy .env.example to .env
        with open(env_example, 'r') as f:
            content = f.read()
        
        with open(env_file, 'w') as f:
            f.write(content)
        
        print("✅ Created .env file from .env.example")
        print("⚠️  Please edit .env file and add your Gemini API key")
    elif env_file.exists():
        print("✅ .env file already exists")
    else:
        print("❌ .env.example file not found")

def install_dependencies():
    """Install Python dependencies"""
    try:
        print("📦 Installing dependencies...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Dependencies installed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing dependencies: {e}")
        return False
    return True

def test_installation():
    """Test if the installation was successful"""
    try:
        print("🧪 Testing installation...")
        
        # Test imports
        import google.generativeai
        import langchain
        import sentence_transformers
        import transformers
        
        print("✅ All required packages imported successfully!")
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def main():
    """Main setup function"""
    print("🚀 Setting up AI Hallucination Detection System")
    print("=" * 50)
    
    # Create directories
    print("\n📁 Creating directories...")
    create_directories()
    
    # Create .env file
    print("\n🔧 Setting up environment...")
    create_env_file()
    
    # Install dependencies
    print("\n📦 Installing dependencies...")
    if not install_dependencies():
        print("❌ Setup failed during dependency installation")
        return False
    
    # Test installation
    print("\n🧪 Testing installation...")
    if not test_installation():
        print("❌ Setup failed during testing")
        return False
    
    print("\n🎉 Setup completed successfully!")
    print("\n📋 Next steps:")
    print("1. Edit .env file and add your Gemini API key")
    print("2. Run: python src/member2_detection/test_detection.py")
    print("3. Run: python src/member2_detection/main.py")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
