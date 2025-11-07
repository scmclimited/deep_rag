"""
Database utility functions for connecting to PostgreSQL.
Centralized DB connection to avoid DRY violations.
"""
import os
import psycopg2
from dotenv import load_dotenv

load_dotenv()


def connect():
    """
    Create and return a PostgreSQL database connection.
    
    Uses environment variables:
    - DB_HOST: Database host (default: localhost)
    - DB_PORT: Database port (default: 5432)
    - DB_USER: Database user
    - DB_PASS: Database password
    - DB_NAME: Database name
    
    Returns:
        psycopg2.connection: Database connection object
    """
    return psycopg2.connect(
        host=os.getenv("DB_HOST"),
        port=os.getenv("DB_PORT"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASS"),
        dbname=os.getenv("DB_NAME")
    )

