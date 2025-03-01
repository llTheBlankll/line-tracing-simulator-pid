"""
Launcher script for the line following robot simulation.
This file runs the main function from the src.main module.
"""
import os
import sys

# Add the project directory to the Python path if needed
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

# Import and run the main function from our modular codebase
try:
    from src.main import main
    main()
except ImportError as e:
    print(f"Error importing modules: {e}")
    print("Make sure you have installed all required packages:")
    print("pip install -r requirements.txt")
except Exception as e:
    print(f"Error running simulation: {e}") 