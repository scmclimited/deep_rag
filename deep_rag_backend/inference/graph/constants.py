"""
Constants for LangGraph pipeline.
"""
import os
from dotenv import load_dotenv
load_dotenv()

MAX_ITERS = int(os.getenv('MAX_ITERS', '5'))  # Increased from 3 to 5 for better convergence on complex multi-document queries
THRESH = float(os.getenv('THRESH', '0.30'))   # matches CE/lex+vec heuristic

