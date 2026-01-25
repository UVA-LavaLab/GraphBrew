#!/usr/bin/env python3
"""
Result file I/O utilities for GraphBrew experiments.

This module provides utilities for:
- Reading and writing experiment results (JSON, CSV)
- Managing intermediate result files between phases
- Aggregating results from multiple runs
- Result validation and schema checking

Example usage:
    from lib.results import ResultsManager
    
    # Create results manager
    results = ResultsManager("/path/to/results")
    
    # Save phase results
    results.save_phase("reorder", {"graph": "web-Stanford", "algorithm": "rcm", ...})
    
    # Load previous results
    benchmarks = results.load_phase("benchmark")
    
    # Aggregate multiple runs
    combined = results.aggregate_results(["run1.json", "run2.json"])
"""

import csv
import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple, Union


# ─────────────────────────────────────────────────────────────────────────────
# File I/O Utilities
# ─────────────────────────────────────────────────────────────────────────────

def read_json(filepath: Union[str, Path]) -> Optional[Dict[str, Any]]:
    """
    Read JSON file safely.
    
    Args:
        filepath: Path to JSON file
        
    Returns:
        Parsed JSON data or None if error
    """
    try:
        with open(filepath, 'r') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError, IOError) as e:
        return None


def write_json(filepath: Union[str, Path], data: Any, 
               indent: int = 2, ensure_dir: bool = True) -> bool:
    """
    Write JSON file safely.
    
    Args:
        filepath: Path to write to
        data: Data to serialize
        indent: JSON indentation
        ensure_dir: Create parent directories if needed
        
    Returns:
        True if successful
    """
    try:
        filepath = Path(filepath)
        if ensure_dir:
            filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=indent, default=str)
        return True
    except (IOError, TypeError) as e:
        return False


def read_csv(filepath: Union[str, Path]) -> Optional[List[Dict[str, str]]]:
    """
    Read CSV file as list of dicts.
    
    Args:
        filepath: Path to CSV file
        
    Returns:
        List of row dictionaries or None if error
    """
    try:
        with open(filepath, 'r', newline='') as f:
            reader = csv.DictReader(f)
            return list(reader)
    except (FileNotFoundError, IOError) as e:
        return None


def write_csv(filepath: Union[str, Path], data: List[Dict[str, Any]],
              fieldnames: List[str] = None, ensure_dir: bool = True) -> bool:
    """
    Write CSV file from list of dicts.
    
    Args:
        filepath: Path to write to
        data: List of row dictionaries
        fieldnames: Column order (auto-detect if None)
        ensure_dir: Create parent directories if needed
        
    Returns:
        True if successful
    """
    if not data:
        return False
    
    try:
        filepath = Path(filepath)
        if ensure_dir:
            filepath.parent.mkdir(parents=True, exist_ok=True)
        
        if fieldnames is None:
            # Collect all keys in order of first appearance
            fieldnames = []
            for row in data:
                for key in row.keys():
                    if key not in fieldnames:
                        fieldnames.append(key)
        
        with open(filepath, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(data)
        return True
    except (IOError, TypeError) as e:
        return False


def append_csv(filepath: Union[str, Path], row: Dict[str, Any]) -> bool:
    """
    Append a single row to CSV file.
    
    Args:
        filepath: Path to CSV file
        row: Row data to append
        
    Returns:
        True if successful
    """
    try:
        filepath = Path(filepath)
        file_exists = filepath.exists()
        
        with open(filepath, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=row.keys())
            if not file_exists:
                writer.writeheader()
            writer.writerow(row)
        return True
    except IOError as e:
        return False


# ─────────────────────────────────────────────────────────────────────────────
# Result File Naming
# ─────────────────────────────────────────────────────────────────────────────

def generate_timestamp() -> str:
    """Generate timestamp string for filenames."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def generate_result_filename(prefix: str, extension: str = "json",
                            timestamp: bool = True) -> str:
    """
    Generate result filename with optional timestamp.
    
    Args:
        prefix: Filename prefix
        extension: File extension
        timestamp: Include timestamp
        
    Returns:
        Generated filename
    """
    if timestamp:
        ts = generate_timestamp()
        return f"{prefix}_{ts}.{extension}"
    return f"{prefix}.{extension}"


def parse_result_filename(filename: str) -> Optional[Dict[str, str]]:
    """
    Parse result filename to extract components.
    
    Expected format: prefix_YYYYMMDD_HHMMSS.ext
    
    Args:
        filename: Filename to parse
        
    Returns:
        Dict with 'prefix', 'timestamp', 'extension' or None
    """
    pattern = r'^(.+?)_(\d{8}_\d{6})\.(\w+)$'
    match = re.match(pattern, filename)
    if match:
        return {
            'prefix': match.group(1),
            'timestamp': match.group(2),
            'extension': match.group(3),
        }
    
    # Try without timestamp
    pattern2 = r'^(.+?)\.(\w+)$'
    match2 = re.match(pattern2, filename)
    if match2:
        return {
            'prefix': match2.group(1),
            'timestamp': None,
            'extension': match2.group(2),
        }
    
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Results Manager
# ─────────────────────────────────────────────────────────────────────────────

class ResultsManager:
    """
    Manager for experiment result files.
    
    Handles:
    - Phase-based result organization
    - Automatic timestamping
    - Result aggregation
    - Format conversion
    """
    
    def __init__(self, results_dir: Union[str, Path], 
                 experiment_name: str = None,
                 use_timestamps: bool = True):
        """
        Initialize results manager.
        
        Args:
            results_dir: Directory for result files
            experiment_name: Optional experiment identifier
            use_timestamps: Add timestamps to filenames
        """
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        self.experiment_name = experiment_name or "experiment"
        self.use_timestamps = use_timestamps
        self.session_id = generate_timestamp() if use_timestamps else None
        
        # Track files written this session
        self.session_files: Dict[str, Path] = {}
    
    def _get_filepath(self, name: str, extension: str = "json") -> Path:
        """Get filepath for a result file."""
        if self.use_timestamps and self.session_id:
            filename = f"{name}_{self.session_id}.{extension}"
        else:
            filename = f"{name}.{extension}"
        return self.results_dir / filename
    
    # ─────────────────────────────────────────────────────────────────────────
    # Phase Results
    # ─────────────────────────────────────────────────────────────────────────
    
    def save_phase(self, phase_name: str, data: Any, 
                   format: str = "json") -> Optional[Path]:
        """
        Save results for a phase.
        
        Args:
            phase_name: Name of the phase
            data: Result data
            format: Output format (json or csv)
            
        Returns:
            Path to saved file or None if error
        """
        filepath = self._get_filepath(phase_name, format)
        
        if format == "json":
            success = write_json(filepath, data)
        elif format == "csv":
            if isinstance(data, list):
                success = write_csv(filepath, data)
            else:
                success = False
        else:
            return None
        
        if success:
            self.session_files[phase_name] = filepath
            return filepath
        return None
    
    def load_phase(self, phase_name: str, 
                   format: str = "json") -> Optional[Any]:
        """
        Load results for a phase (from current session or latest).
        
        Args:
            phase_name: Name of the phase
            format: File format
            
        Returns:
            Loaded data or None
        """
        # Try current session first
        if phase_name in self.session_files:
            filepath = self.session_files[phase_name]
            if filepath.exists():
                if format == "json":
                    return read_json(filepath)
                elif format == "csv":
                    return read_csv(filepath)
        
        # Try to find latest file
        filepath = self.find_latest(phase_name, format)
        if filepath:
            if format == "json":
                return read_json(filepath)
            elif format == "csv":
                return read_csv(filepath)
        
        return None
    
    def find_latest(self, prefix: str, extension: str = "json") -> Optional[Path]:
        """
        Find the most recent result file with given prefix.
        
        Args:
            prefix: Filename prefix
            extension: File extension
            
        Returns:
            Path to latest file or None
        """
        pattern = f"{prefix}*.{extension}"
        files = list(self.results_dir.glob(pattern))
        
        if not files:
            return None
        
        # Sort by modification time
        files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        return files[0]
    
    def find_all(self, prefix: str, extension: str = "json") -> List[Path]:
        """
        Find all result files with given prefix.
        
        Args:
            prefix: Filename prefix
            extension: File extension
            
        Returns:
            List of matching paths (newest first)
        """
        pattern = f"{prefix}*.{extension}"
        files = list(self.results_dir.glob(pattern))
        files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        return files
    
    # ─────────────────────────────────────────────────────────────────────────
    # Aggregation
    # ─────────────────────────────────────────────────────────────────────────
    
    def aggregate_json(self, filepaths: List[Union[str, Path]], 
                       key: str = None) -> Dict[str, Any]:
        """
        Aggregate multiple JSON result files.
        
        Args:
            filepaths: List of JSON files to aggregate
            key: If set, aggregate by this key
            
        Returns:
            Aggregated data
        """
        results = []
        for fp in filepaths:
            data = read_json(fp)
            if data:
                results.append(data)
        
        if not results:
            return {}
        
        if key and all(isinstance(r, list) for r in results):
            # Merge lists, keyed by field
            aggregated = {}
            for result_list in results:
                for item in result_list:
                    if isinstance(item, dict) and key in item:
                        item_key = item[key]
                        if item_key not in aggregated:
                            aggregated[item_key] = item
            return {'results': list(aggregated.values())}
        
        elif all(isinstance(r, list) for r in results):
            # Simple list concatenation
            combined = []
            for result_list in results:
                combined.extend(result_list)
            return {'results': combined}
        
        elif all(isinstance(r, dict) for r in results):
            # Merge dictionaries
            merged = {}
            for result_dict in results:
                merged.update(result_dict)
            return merged
        
        return {'results': results}
    
    def aggregate_csv(self, filepaths: List[Union[str, Path]]) -> List[Dict]:
        """
        Aggregate multiple CSV files into one list.
        
        Args:
            filepaths: List of CSV files
            
        Returns:
            Combined list of rows
        """
        combined = []
        for fp in filepaths:
            data = read_csv(fp)
            if data:
                combined.extend(data)
        return combined
    
    # ─────────────────────────────────────────────────────────────────────────
    # Conversion
    # ─────────────────────────────────────────────────────────────────────────
    
    def json_to_csv(self, json_path: Union[str, Path],
                    csv_path: Union[str, Path] = None,
                    flatten: bool = True) -> Optional[Path]:
        """
        Convert JSON results to CSV.
        
        Args:
            json_path: Path to JSON file
            csv_path: Output CSV path (auto-generate if None)
            flatten: Flatten nested structures
            
        Returns:
            Path to CSV file or None
        """
        data = read_json(json_path)
        if data is None:
            return None
        
        # Handle different JSON structures
        if isinstance(data, list):
            rows = data
        elif isinstance(data, dict) and 'results' in data:
            rows = data['results']
        elif isinstance(data, dict):
            rows = [data]
        else:
            return None
        
        if flatten:
            rows = [flatten_dict(row) for row in rows if isinstance(row, dict)]
        
        if csv_path is None:
            json_path = Path(json_path)
            csv_path = json_path.with_suffix('.csv')
        
        if write_csv(csv_path, rows):
            return Path(csv_path)
        return None
    
    def csv_to_json(self, csv_path: Union[str, Path],
                    json_path: Union[str, Path] = None) -> Optional[Path]:
        """
        Convert CSV results to JSON.
        
        Args:
            csv_path: Path to CSV file
            json_path: Output JSON path (auto-generate if None)
            
        Returns:
            Path to JSON file or None
        """
        data = read_csv(csv_path)
        if data is None:
            return None
        
        if json_path is None:
            csv_path = Path(csv_path)
            json_path = csv_path.with_suffix('.json')
        
        if write_json(json_path, data):
            return Path(json_path)
        return None
    
    # ─────────────────────────────────────────────────────────────────────────
    # Session Management
    # ─────────────────────────────────────────────────────────────────────────
    
    def get_session_files(self) -> Dict[str, Path]:
        """Get all files written in current session."""
        return self.session_files.copy()
    
    def save_session_manifest(self) -> Optional[Path]:
        """Save manifest of current session files."""
        manifest = {
            'session_id': self.session_id,
            'experiment_name': self.experiment_name,
            'timestamp': datetime.now().isoformat(),
            'files': {k: str(v) for k, v in self.session_files.items()}
        }
        
        filepath = self._get_filepath('session_manifest', 'json')
        if write_json(filepath, manifest):
            return filepath
        return None
    
    def list_sessions(self) -> List[Dict[str, Any]]:
        """List all session manifests."""
        sessions = []
        for manifest_file in self.results_dir.glob('session_manifest_*.json'):
            data = read_json(manifest_file)
            if data:
                data['manifest_file'] = str(manifest_file)
                sessions.append(data)
        
        sessions.sort(key=lambda x: x.get('timestamp', ''), reverse=True)
        return sessions


# ─────────────────────────────────────────────────────────────────────────────
# Utility Functions
# ─────────────────────────────────────────────────────────────────────────────

def flatten_dict(d: Dict[str, Any], parent_key: str = '', 
                 sep: str = '.') -> Dict[str, Any]:
    """
    Flatten nested dictionary.
    
    Args:
        d: Dictionary to flatten
        parent_key: Prefix for keys
        sep: Separator for nested keys
        
    Returns:
        Flattened dictionary
    """
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep).items())
        elif isinstance(v, list):
            # Convert lists to string representation
            items.append((new_key, str(v)))
        else:
            items.append((new_key, v))
    return dict(items)


def filter_results(results: List[Dict], 
                   filters: Dict[str, Any]) -> List[Dict]:
    """
    Filter result list by criteria.
    
    Args:
        results: List of result dicts
        filters: Filter criteria {field: value}
        
    Returns:
        Filtered results
    """
    filtered = []
    for result in results:
        match = True
        for field, value in filters.items():
            if field not in result:
                match = False
                break
            if callable(value):
                if not value(result[field]):
                    match = False
                    break
            elif result[field] != value:
                match = False
                break
        if match:
            filtered.append(result)
    return filtered


def group_results(results: List[Dict], 
                  group_by: str) -> Dict[str, List[Dict]]:
    """
    Group results by a field.
    
    Args:
        results: List of result dicts
        group_by: Field to group by
        
    Returns:
        Dict mapping group values to results
    """
    groups = {}
    for result in results:
        key = result.get(group_by, 'unknown')
        if key not in groups:
            groups[key] = []
        groups[key].append(result)
    return groups


def compute_statistics(values: List[float]) -> Dict[str, float]:
    """
    Compute basic statistics for a list of values.
    
    Args:
        values: List of numeric values
        
    Returns:
        Dict with min, max, mean, median, std
    """
    if not values:
        return {}
    
    values = sorted(values)
    n = len(values)
    mean = sum(values) / n
    
    if n % 2 == 0:
        median = (values[n//2 - 1] + values[n//2]) / 2
    else:
        median = values[n//2]
    
    variance = sum((x - mean) ** 2 for x in values) / n
    std = variance ** 0.5
    
    return {
        'min': min(values),
        'max': max(values),
        'mean': mean,
        'median': median,
        'std': std,
        'count': n,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Result Schema Validation
# ─────────────────────────────────────────────────────────────────────────────

class ResultSchema:
    """Simple schema validation for results."""
    
    def __init__(self, required_fields: List[str] = None,
                 optional_fields: List[str] = None,
                 field_types: Dict[str, type] = None):
        """
        Initialize schema.
        
        Args:
            required_fields: List of required field names
            optional_fields: List of optional field names
            field_types: Dict mapping field names to expected types
        """
        self.required = required_fields or []
        self.optional = optional_fields or []
        self.types = field_types or {}
    
    def validate(self, result: Dict) -> Tuple[bool, List[str]]:
        """
        Validate a result against schema.
        
        Args:
            result: Result dict to validate
            
        Returns:
            (is_valid, list of error messages)
        """
        errors = []
        
        # Check required fields
        for field in self.required:
            if field not in result:
                errors.append(f"Missing required field: {field}")
        
        # Check types
        for field, expected_type in self.types.items():
            if field in result and not isinstance(result[field], expected_type):
                actual_type = type(result[field]).__name__
                errors.append(f"Field '{field}' has type {actual_type}, expected {expected_type.__name__}")
        
        return len(errors) == 0, errors


# Common schemas for GraphBrew results
BENCHMARK_RESULT_SCHEMA = ResultSchema(
    required_fields=['graph', 'algorithm', 'benchmark', 'time'],
    optional_fields=['speedup', 'iterations', 'trial', 'timestamp'],
    field_types={'time': (int, float), 'speedup': (int, float, type(None))}
)

REORDER_RESULT_SCHEMA = ResultSchema(
    required_fields=['graph', 'algorithm', 'success'],
    optional_fields=['time', 'output_path', 'error'],
    field_types={'success': bool}
)

CACHE_RESULT_SCHEMA = ResultSchema(
    required_fields=['graph', 'algorithm', 'benchmark', 'hit_rate'],
    optional_fields=['l1_hit_rate', 'l2_hit_rate', 'l3_hit_rate'],
    field_types={'hit_rate': (int, float)}
)


__all__ = [
    'read_json',
    'write_json', 
    'read_csv',
    'write_csv',
    'append_csv',
    'generate_timestamp',
    'generate_result_filename',
    'parse_result_filename',
    'ResultsManager',
    'flatten_dict',
    'filter_results',
    'group_results',
    'compute_statistics',
    'ResultSchema',
    'BENCHMARK_RESULT_SCHEMA',
    'REORDER_RESULT_SCHEMA',
    'CACHE_RESULT_SCHEMA',
]
