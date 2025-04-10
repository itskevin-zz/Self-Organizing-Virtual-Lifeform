import json
from threading import Lock
from typing import List, Dict, Any, Optional
import os

class ThreadSafeLogger:
    """
    Thread-safe logger for writing and reading JSONL files.
    """
    def __init__(self, filename: str = "log.jsonl"):
        self.filename = filename
        self.lock = Lock()
        self._ensure_file_exists()

    def _ensure_file_exists(self) -> None:
        """Ensure the log file exists."""
        if not os.path.exists(self.filename):
            with open(self.filename, 'w') as f:
                pass

    def write(self, data: Dict[str, Any]) -> None:
        """
        Write data to the log file in a thread-safe manner.
        
        Args:
            data: Dictionary containing the data to log
        """
        with self.lock:
            with open(self.filename, 'a') as f:
                json.dump(data, f)
                f.write('\n')

    def read(self) -> List[Dict[str, Any]]:
        """
        Read all entries from the log file in a thread-safe manner.
        
        Returns:
            List of dictionaries containing the logged data
        """
        with self.lock:
            try:
                with open(self.filename, 'r') as f:
                    return [json.loads(line) for line in f if line.strip()]
            except (json.JSONDecodeError, IOError):
                return []

    def clear(self) -> None:
        """
        Clear the log file in a thread-safe manner.
        """
        with self.lock:
            with open(self.filename, 'w') as f:
                pass

    def get_latest_entry(self) -> Optional[Dict[str, Any]]:
        """
        Get the most recent entry from the log file.
        
        Returns:
            The most recent log entry or None if the log is empty
        """
        entries = self.read()
        return entries[-1] if entries else None

    def get_entries_since(self, timestamp: float) -> List[Dict[str, Any]]:
        """
        Get all entries since a given timestamp.
        
        Args:
            timestamp: Unix timestamp to filter entries
            
        Returns:
            List of entries newer than the given timestamp
        """
        entries = self.read()
        return [entry for entry in entries if entry.get('timestamp', 0) > timestamp] 