import json
from threading import Lock
from typing import Any, Dict, List, Optional
from pathlib import Path
from datetime import datetime

class ThreadSafeLogger:
    """
    A thread-safe logger for writing and reading JSONL log files.
    """
    def __init__(self, filename: str = "log.jsonl"):
        """
        Initialize the logger with a filename.
        
        Args:
            filename: Name of the log file
        """
        self.filename = filename
        self.lock = Lock()
        self._ensure_log_file()
    
    def _ensure_log_file(self) -> None:
        """Ensure the log file exists."""
        if not Path(self.filename).exists():
            with open(self.filename, "w") as f:
                pass
    
    def write(self, data: Dict[str, Any]) -> None:
        """
        Write a log entry to the file in a thread-safe manner.
        
        Args:
            data: Dictionary containing the log data
        """
        with self.lock:
            try:
                # Add timestamp if not present
                if "timestamp" not in data:
                    data["timestamp"] = datetime.now().isoformat()
                
                with open(self.filename, "a") as f:
                    json.dump(data, f)
                    f.write("\n")
            except Exception as e:
                print(f"Error writing to log: {e}")
    
    def read(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Read all log entries from the file in a thread-safe manner.
        
        Returns:
            List of log entries as dictionaries
        """
        with self.lock:
            try:
                entries = []
                with open(self.filename, "r") as f:
                    for line in f:
                        try:
                            entry = json.loads(line)
                            entries.append(entry)
                            if limit and len(entries) >= limit:
                                break
                        except json.JSONDecodeError:
                            continue
                return entries
            except FileNotFoundError:
                return []
    
    def clear(self) -> None:
        """
        Clear all log entries from the file in a thread-safe manner.
        """
        with self.lock:
            try:
                with open(self.filename, "w") as f:
                    pass
            except Exception as e:
                print(f"Error clearing log: {e}")
    
    def get_latest_entry(self) -> Optional[Dict[str, Any]]:
        """
        Get the most recent log entry.
        
        Returns:
            The latest log entry as a dictionary, or None if no entries exist
        """
        entries = self.read()
        return entries[-1] if entries else None
    
    def get_entries_by_type(self, entry_type: str) -> List[Dict[str, Any]]:
        """
        Get all log entries of a specific type.
        
        Args:
            entry_type: Type of log entries to retrieve
            
        Returns:
            List of matching log entries
        """
        return [entry for entry in self.read() if entry.get('type') == entry_type]
    
    def log_error(self, message: str, context: Optional[Dict[str, Any]] = None) -> None:
        """Log an error message with optional context."""
        entry = {
            "type": "error",
            "message": message,
            "context": context or {}
        }
        self.write(entry)
    
    def log_info(self, message: str, context: Optional[Dict[str, Any]] = None) -> None:
        """Log an info message with optional context."""
        entry = {
            "type": "info",
            "message": message,
            "context": context or {}
        }
        self.write(entry)
    
    def log_interaction(self, prompt: str, response: str, confidence: Optional[float] = None) -> None:
        """Log a user interaction."""
        entry = {
            "type": "interaction",
            "prompt": prompt,
            "response": response,
            "confidence": confidence
        }
        self.write(entry)

# Create a singleton instance
logger = ThreadSafeLogger() 