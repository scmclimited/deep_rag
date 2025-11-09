"""
Query sanitization for PostgreSQL tsquery.
"""
import re


def sanitize_query_for_tsquery(query: str) -> str:
    """
    Sanitize query string for PostgreSQL tsquery to prevent syntax errors.
    
    Handles special characters that break tsquery syntax:
    - Replaces literal & with "and" (to avoid confusion with tsquery AND operator)
    - Removes/escapes other tsquery operators: |, !, (, ), :, *
    - Removes leading/trailing special characters
    - Strips bullet points and other formatting characters
    
    Args:
        query: Raw query string (potentially from LLM output)
        
    Returns:
        Sanitized query string safe for tsquery
    """
    # Normalize line breaks and strip leading bullet characters / whitespace
    query = query.replace('\r', ' ').replace('\n', ' ')
    query = re.sub(r'^[\*\-\•\s]+', '', query.strip())

    # Replace literal & with "and" (preserve the meaning but avoid tsquery syntax conflicts)
    query = query.replace('&', ' and ')

    # Remove all characters that could break tsquery syntax, leaving only letters, numbers, and spaces
    query = re.sub(r'[^0-9a-zA-Z\s]', ' ', query)

    # Collapse multiple spaces and trim
    query = re.sub(r'\s+', ' ', query).strip()

    return query

