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
    # Remove leading bullet points, asterisks, dashes
    query = re.sub(r'^[\*\-\•\s]+', '', query.strip())
    
    # Replace literal & with "and" (preserve the meaning but avoid tsquery syntax conflicts)
    query = query.replace('&', ' and ')
    
    # Remove or escape other tsquery special characters
    # These characters have special meaning in tsquery: |, !, (, ), :, *
    # We'll remove them to avoid syntax errors, as they're not typically needed for basic search
    query = re.sub(r'[\!\|\:\*]', ' ', query)
    
    # Remove quotes that might cause issues
    query = query.replace('"', '').replace("'", '')
    
    # Normalize whitespace
    query = re.sub(r'\s+', ' ', query).strip()
    
    return query

