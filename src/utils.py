"""
Utility functions
"""
from datetime import datetime

# Log message with timestamp
def log(message:str):
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}")