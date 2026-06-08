"""
INSTALL ALL DEPENDENCIES
Run this once to set up everything
"""

import subprocess

import sys

import os

print("=" * 70)

print("INSTALLING ALL DEPENDENCIES")

print("=" * 70)

             

print("\n1. Upgrading pip...")

subprocess.run([sys.executable, "-m", "pip", "install", "--upgrade", "pip"])

                                

print("\n2. Installing LangChain dependencies...")

subprocess.run([sys.executable, "-m", "pip", "install", 

                "langchain", "langchain-ollama", "langchain-core"])

                           

print("\n3. Installing SDXL dependencies...")

subprocess.run([sys.executable, "-m", "pip", "install",

                "torch", "torchvision", "diffusers", "transformers",

                "accelerate", "safetensors", "pillow", "numpy"])

                   

print("\n4. Installing utilities...")

subprocess.run([sys.executable, "-m", "pip", "install",

                "pyyaml", "opencv-python"])

print("\n" + "=" * 70)

print("[SUCCESS] INSTALLATION COMPLETE!")

print("=" * 70)

print("\nNext steps:")

print("   1. Install Ollama from: https://ollama.com/download")

print("   2. Run: ollama serve (in a new terminal)")

print("   3. Run: ollama pull llama3.2")

print("   4. Run: python run_everything.py")

print("=" * 70)

