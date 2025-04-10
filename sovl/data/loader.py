import json
from typing import List, Dict, Any, Optional
import os

class InsufficientDataError(Exception):
    """Raised when there is insufficient data for processing."""
    pass

def load_jsonl(file_path: str, min_entries: int = 10) -> List[Dict[str, str]]:
    """
    Load and validate data from a JSONL file.
    
    Args:
        file_path: Path to the JSONL file
        min_entries: Minimum number of entries required
        
    Returns:
        List of dictionaries containing prompt and completion pairs
        
    Raises:
        InsufficientDataError: If the file contains fewer than min_entries valid entries
        IOError: If the file cannot be read
    """
    data = []
    error_log = []

    try:
        # Attempt to open and read the file
        with open(file_path, 'r') as file:
            for line_number, line in enumerate(file, start=1):
                try:
                    entry = json.loads(line.strip())
                    # Validate the structure of each entry
                    if not isinstance(entry.get("prompt"), str) or not isinstance(entry.get("response"), str):
                        error_log.append(f"Line {line_number}: Missing or invalid 'prompt' or 'response'. Skipping.")
                        continue
                    # Append valid entry
                    data.append({"prompt": entry["prompt"], "completion": entry["response"]})
                except json.JSONDecodeError as e:
                    error_log.append(f"Line {line_number}: JSON decode error: {e}. Skipping.")
        
        # Print warnings if any
        if error_log:
            print("Warnings encountered during data loading:")
            for error in error_log:
                print(f"WARNING: {error}")
            # Optionally, write errors to a file
            with open("data_load_errors.log", "w") as log_file:
                log_file.write("\n".join(error_log))
        
        # Check if the minimum threshold is met
        if len(data) < min_entries:
            raise InsufficientDataError(
                f"File contains only {len(data)} valid entries, but minimum required is {min_entries}"
            )
        
        return data

    except IOError as e:
        raise IOError(f"Error reading file {file_path}: {e}")

def validate_data_format(data: List[Dict[str, str]]) -> bool:
    """
    Validate the format of loaded data.
    
    Args:
        data: List of data entries to validate
        
    Returns:
        True if data format is valid, False otherwise
    """
    if not isinstance(data, list):
        return False
    
    for entry in data:
        if not isinstance(entry, dict):
            return False
        if not all(key in entry for key in ["prompt", "completion"]):
            return False
        if not all(isinstance(entry[key], str) for key in ["prompt", "completion"]):
            return False
    
    return True

def split_data(data: List[Dict[str, str]], train_ratio: float = 0.8) -> tuple[List[Dict[str, str]], List[Dict[str, str]]]:
    """
    Split data into training and validation sets.
    
    Args:
        data: List of data entries to split
        train_ratio: Ratio of data to use for training
        
    Returns:
        Tuple of (training_data, validation_data)
    """
    split_idx = int(len(data) * train_ratio)
    return data[:split_idx], data[split_idx:] 