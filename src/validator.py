"""
Data Validation Module
Validates incoming datasets for schema completeness, encoding, and data integrity.
"""
from typing import Dict, Any, List


def validate_schema(data_record: Dict[str, Any], required_fields: List[str]) -> bool:
    """
    Validates that a data record contains all required fields and non-null values.

    Args:
        data_record (Dict[str, Any]): Dictionary representing a dataset row.
        required_fields (List[str]): List of essential column names.

    Returns:
        bool: True if valid, raises ValueError otherwise.
    """
    missing_fields = [field for field in required_fields if field not in data_record or data_record[field] is None]
    if missing_fields:
        raise ValueError(f"Schema validation failed. Missing required fields: {missing_fields}")
    return True
