import json
from typing import List, Dict, Any, Optional
from pathlib import Path
from ..core.system_logging import logger

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
        FileNotFoundError: If the file does not exist
        json.JSONDecodeError: If the file contains invalid JSON
    """
    data = []
    error_log = []

    try:
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
                f"File contains only {len(data)} valid entries, minimum required is {min_entries}"
            )
            
        return data
        
    except FileNotFoundError:
        raise FileNotFoundError(f"Data file not found: {file_path}")
    except json.JSONDecodeError as e:
        raise json.JSONDecodeError(f"Invalid JSON in data file: {e}", e.doc, e.pos)

def validate_data_format(data: List[Dict[str, str]]) -> bool:
    """
    Validate the format of loaded data.
    
    Args:
        data: List of data entries to validate
        
    Returns:
        True if data is valid, False otherwise
    """
    if not isinstance(data, list):
        return False
        
    for entry in data:
        if not isinstance(entry, dict):
            return False
        if "prompt" not in entry or "completion" not in entry:
            return False
        if not isinstance(entry["prompt"], str) or not isinstance(entry["completion"], str):
            return False
            
    return True

def save_data(data: List[Dict[str, str]], file_path: str) -> None:
    """
    Save data to a JSONL file.
    
    Args:
        data: List of data entries to save
        file_path: Path where to save the data
    """
    with open(file_path, 'w') as f:
        for entry in data:
            json.dump(entry, f)
            f.write('\n')

# Create a singleton instance
data_loader = DataLoader() 