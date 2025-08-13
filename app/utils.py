"""
Utility functions for file processing and data handling.
"""
import os
import json
import csv
import zipfile
import tempfile
import mimetypes
import re
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import pandas as pd
from fastapi import UploadFile
import aiofiles
import shutil
import requests
from bs4 import BeautifulSoup

# Supported file types and their extensions
SUPPORTED_EXTENSIONS = {
    '.csv', '.json', '.txt', '.html', '.htm', '.xml', 
    '.zip', '.xlsx', '.xls', '.tsv', '.md', '.log'
}

SUPPORTED_MIME_TYPES = {
    'text/csv', 'application/json', 'text/plain', 'text/html', 
    'application/xml', 'text/xml', 'application/zip',
    'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
    'application/vnd.ms-excel', 'text/tab-separated-values',
    'text/markdown', 'application/octet-stream'
}

def validate_file_type(filename: str) -> bool:
    """
    Validate if the uploaded file type is supported.
    
    Args:
        filename: Name of the uploaded file
        
    Returns:
        True if file type is supported, False otherwise
    """
    if not filename:
        return False
    
    # Check extension
    extension = Path(filename).suffix.lower()
    if extension in SUPPORTED_EXTENSIONS:
        return True
    
    # Check MIME type
    mime_type, _ = mimetypes.guess_type(filename)
    if mime_type in SUPPORTED_MIME_TYPES:
        return True
    
    return False

def create_sandbox_directory(request_id: str) -> Path:
    """
    Create a sandbox directory for a specific request.
    
    Args:
        request_id: Unique request identifier
        
    Returns:
        Path to the created sandbox directory
    """
    sandbox_path = Path("sandbox") / request_id
    sandbox_path.mkdir(parents=True, exist_ok=True)
    return sandbox_path

async def save_uploaded_files(files: List[UploadFile], sandbox_path: Path) -> List[Path]:
    """
    Save uploaded files to the sandbox directory with smart naming for data files.
    
    Args:
        files: List of uploaded files
        sandbox_path: Path to sandbox directory
        
    Returns:
        List of paths to saved files
    """
    saved_files = []
    
    for i, file in enumerate(files):
        # Use smart naming for JSON files to ensure consistency
        if file.filename and file.filename.lower().endswith('.json'):
            safe_filename = "data.json"  # Standardize JSON file name
        else:
            # Generate safe filename for other files
            safe_filename = sanitize_filename(file.filename or f"file_{i}")
        
        file_path = sandbox_path / safe_filename
        
        # Save file asynchronously
        async with aiofiles.open(file_path, 'wb') as f:
            content = await file.read()
            await f.write(content)
        
        saved_files.append(file_path)
        
        # Handle ZIP files by extracting them
        if file_path.suffix.lower() == '.zip':
            extracted_files = extract_zip_file(file_path, sandbox_path)
            saved_files.extend(extracted_files)
    
    return saved_files

def sanitize_filename(filename: str) -> str:
    """
    Sanitize filename to prevent path traversal and invalid characters.
    
    Args:
        filename: Original filename
        
    Returns:
        Sanitized filename
    """
    # Remove directory separators and invalid characters
    invalid_chars = '<>:"/\\|?*'
    for char in invalid_chars:
        filename = filename.replace(char, '_')
    
    # Remove leading/trailing spaces and dots
    filename = filename.strip(' .')
    
    # Ensure filename is not empty
    if not filename:
        filename = "unnamed_file"
    
    # Limit length
    if len(filename) > 255:
        name, ext = os.path.splitext(filename)
        filename = name[:250] + ext
    
    return filename

def extract_zip_file(zip_path: Path, extract_to: Path) -> List[Path]:
    """
    Extract ZIP file contents to the specified directory.
    
    Args:
        zip_path: Path to ZIP file
        extract_to: Directory to extract files to
        
    Returns:
        List of extracted file paths
    """
    extracted_files = []
    
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            for member in zip_ref.namelist():
                # Skip directories and hidden files
                if member.endswith('/') or member.startswith('.'):
                    continue
                
                # Sanitize member name to prevent path traversal
                safe_name = sanitize_filename(os.path.basename(member))
                if not safe_name:
                    continue
                
                # Extract file
                extract_path = extract_to / f"extracted_{safe_name}"
                with zip_ref.open(member) as source, open(extract_path, 'wb') as target:
                    shutil.copyfileobj(source, target)
                
                extracted_files.append(extract_path)
                
    except zipfile.BadZipFile:
        pass  # Ignore bad zip files
    except Exception:
        pass  # Ignore other extraction errors
    
    return extracted_files

def analyze_file_structure(file_paths: List[Path]) -> Dict[str, Any]:
    """
    Analyze the structure and content of uploaded files.
    
    Args:
        file_paths: List of file paths to analyze
        
    Returns:
        Dictionary containing file analysis results
    """
    analysis = {
        'total_files': len(file_paths),
        'file_types': {},
        'files': []
    }
    
    for file_path in file_paths:
        file_info = analyze_single_file(file_path)
        analysis['files'].append(file_info)
        
        # Count file types
        file_type = file_info['type']
        analysis['file_types'][file_type] = analysis['file_types'].get(file_type, 0) + 1
    
    return analysis

def get_all_available_files(sandbox_path: Path, original_files: List[Path]) -> List[Path]:
    """
    Get all files available for analysis, including original files and any scraped data files.
    
    Args:
        sandbox_path: Path to the sandbox directory
        original_files: List of originally uploaded file paths
        
    Returns:
        List of all available file paths for analysis
    """
    all_files = list(original_files)
    
    # Check for common scraped data file names
    scraped_data_files = [
        'scraped_data.csv',
        'scraped_data.json',
        'scraped_data.txt',
        'scraped_data.xlsx',
        'scraped_data.html'
    ]
    
    # Add any existing scraped data files in the sandbox to all_files
    for scraped_file in scraped_data_files:
        path = sandbox_path / scraped_file
        if path.exists():
            all_files.append(path)
    
    return all_files

def analyze_single_file(file_path: Path) -> Dict[str, Any]:
    """
    Analyze a single file and extract metadata.
    
    Args:
        file_path: Path to the file
        
    Returns:
        Dictionary containing file information
    """
    file_info = {
        'name': file_path.name,
        'path': str(file_path),
        'size': file_path.stat().st_size if file_path.exists() else 0,
        'extension': file_path.suffix.lower(),
        'type': 'unknown',
        'encoding': 'unknown',
        'structure': {}
    }
    
    try:
        # Determine file type and analyze structure
        if file_path.suffix.lower() == '.csv':
            file_info.update(analyze_csv_file(file_path))
        elif file_path.suffix.lower() == '.json':
            file_info.update(analyze_json_file(file_path))
        elif file_path.suffix.lower() in ['.xlsx', '.xls']:
            file_info.update(analyze_excel_file(file_path))
        elif file_path.suffix.lower() in ['.html', '.htm']:
            file_info.update(analyze_html_file(file_path))
        elif file_path.suffix.lower() == '.txt':
            file_info.update(analyze_text_file(file_path))
        else:
            file_info['type'] = 'text'
            
    except Exception as e:
        file_info['error'] = str(e)
    
    return file_info

def analyze_csv_file(file_path: Path) -> Dict[str, Any]:
    """Analyze CSV file structure."""
    try:
        # Try to read with pandas to get basic info
        df = pd.read_csv(file_path, nrows=5)  # Read only first 5 rows for analysis
        
        return {
            'type': 'csv',
            'structure': {
                'columns': list(df.columns),
                'num_columns': len(df.columns),
                'estimated_rows': len(df),  # This is just the sample
                'dtypes': df.dtypes.astype(str).to_dict(),
                'sample_data': df.head(2).to_dict('records')
            }
        }
    except Exception:
        return {'type': 'csv', 'structure': {'error': 'Could not parse CSV'}}

def analyze_json_file(file_path: Path) -> Dict[str, Any]:
    """Analyze JSON file structure with enhanced intelligence."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        structure = {
            'data_type': type(data).__name__,
            'estimated_records': 0,
            'complexity': 'simple',
            'nesting_depth': 0,
            'suggested_analysis': 'general'
        }
        
        def calculate_depth(obj, current_depth=0):
            if isinstance(obj, dict):
                return max([calculate_depth(v, current_depth + 1) for v in obj.values()] + [current_depth])
            elif isinstance(obj, list) and obj:
                return max([calculate_depth(item, current_depth) for item in obj[:5]] + [current_depth])  # Sample first 5
            return current_depth
        
        if isinstance(data, list):
            structure['length'] = len(data)
            structure['estimated_records'] = len(data)
            if data:
                structure['item_type'] = type(data[0]).__name__
                if isinstance(data[0], dict):
                    structure['keys'] = list(data[0].keys())
                    structure['nesting_depth'] = calculate_depth(data[0])
                    
                    # Determine complexity and suggested analysis
                    if len(structure['keys']) > 10:
                        structure['complexity'] = 'complex'
                    elif structure['nesting_depth'] > 2:
                        structure['complexity'] = 'nested'
                    
                    # Suggest analysis type based on field names
                    key_names = ' '.join(structure['keys']).lower()
                    if any(time_word in key_names for time_word in ['date', 'time', 'year', 'month', 'timestamp']):
                        structure['suggested_analysis'] = 'timeseries'
                    elif any(stat_word in key_names for stat_word in ['price', 'value', 'amount', 'score', 'rate']):
                        structure['suggested_analysis'] = 'statistical'
                    else:
                        structure['suggested_analysis'] = 'json'
                        
        elif isinstance(data, dict):
            structure['keys'] = list(data.keys())
            structure['nesting_depth'] = calculate_depth(data)
            structure['estimated_records'] = 1
            
            # For dictionary data, estimate records from values
            for key, value in data.items():
                if isinstance(value, list):
                    structure['estimated_records'] = max(structure['estimated_records'], len(value))
                    
            if structure['nesting_depth'] > 1:
                structure['complexity'] = 'nested'
                structure['suggested_analysis'] = 'json'
        
        return {
            'type': 'json',
            'structure': structure
        }
    except UnicodeDecodeError:
        # Try different encodings
        for encoding in ['latin-1', 'cp1252', 'utf-16']:
            try:
                with open(file_path, 'r', encoding=encoding) as f:
                    data = json.load(f)
                return {
                    'type': 'json',
                    'structure': {'encoding_used': encoding, 'data_type': type(data).__name__}
                }
            except:
                continue
        return {'type': 'json', 'structure': {'error': 'Encoding issues'}}
    except json.JSONDecodeError as e:
        return {'type': 'json', 'structure': {'error': f'Invalid JSON: {str(e)}'}}
    except Exception as e:
        return {'type': 'json', 'structure': {'error': f'Could not parse JSON: {str(e)}'}}

def analyze_excel_file(file_path: Path) -> Dict[str, Any]:
    """Analyze Excel file structure."""
    try:
        # Get sheet names
        excel_file = pd.ExcelFile(file_path)
        sheets = excel_file.sheet_names
        
        structure = {
            'sheets': sheets,
            'num_sheets': len(sheets)
        }
        
        # Analyze first sheet
        if sheets:
            df = pd.read_excel(file_path, sheet_name=sheets[0], nrows=5)
            structure['first_sheet'] = {
                'columns': list(df.columns),
                'num_columns': len(df.columns)
            }
        
        return {
            'type': 'excel',
            'structure': structure
        }
    except Exception:
        return {'type': 'excel', 'structure': {'error': 'Could not parse Excel'}}

def analyze_html_file(file_path: Path) -> Dict[str, Any]:
    """Analyze HTML file structure."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Basic HTML analysis (can be enhanced with BeautifulSoup)
        structure = {
            'size_chars': len(content),
            'has_tables': '<table' in content.lower(),
            'has_forms': '<form' in content.lower(),
            'has_scripts': '<script' in content.lower()
        }
        
        return {
            'type': 'html',
            'structure': structure
        }
    except Exception:
        return {'type': 'html', 'structure': {'error': 'Could not parse HTML'}}

def analyze_text_file(file_path: Path) -> Dict[str, Any]:
    """Analyze text file structure."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        lines = content.split('\n')
        structure = {
            'size_chars': len(content),
            'num_lines': len(lines),
            'encoding': 'utf-8'
        }
        
        return {
            'type': 'text',
            'structure': structure
        }
    except Exception:
        return {'type': 'text', 'structure': {'error': 'Could not parse text file'}}

def create_code_file(code: str, sandbox_path: Path) -> Path:
    """
    Create a Python file containing the generated code.
    
    Args:
        code: Python code to save
        sandbox_path: Path to sandbox directory
        
    Returns:
        Path to the created Python file
    """
    code_file = sandbox_path / "analysis.py"
    
    with open(code_file, 'w', encoding='utf-8') as f:
        f.write(code)
    
    return code_file

def read_execution_results(sandbox_path: Path) -> Dict[str, Any]:
    """
    Read execution results from the sandbox directory.
    
    Args:
        sandbox_path: Path to sandbox directory
        
    Returns:
        Dictionary containing execution results
    """
    results = {}
    
    # Read JSON results
    result_file = sandbox_path / "result.json"
    if result_file.exists():
        try:
            with open(result_file, 'r', encoding='utf-8') as f:
                results['json'] = json.load(f)
        except Exception as e:
            results['json_error'] = str(e)
    
    # Read image results (base64 encoded PNGs)
    image_files = list(sandbox_path.glob("*.png"))
    if image_files:
        results['images'] = []
        for img_file in image_files:
            try:
                import base64
                # Get file size before encoding
                file_size = img_file.stat().st_size
                
                with open(img_file, 'rb') as f:
                    img_data = base64.b64encode(f.read()).decode('utf-8')
                    results['images'].append({
                        'filename': img_file.name,
                        'data': img_data,
                        'original_size_bytes': file_size
                    })
            except Exception as e:
                results[f'image_error_{img_file.name}'] = str(e)
    
    # Read text outputs
    for filename in ['stdout.txt', 'stderr.txt', 'output.txt']:
        file_path = sandbox_path / filename
        if file_path.exists():
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    results[filename.replace('.txt', '')] = f.read()
            except Exception as e:
                results[f'{filename}_error'] = str(e)
    
    return results

def get_file_sample_content(file_path: Path, max_lines: int = 10) -> str:
    """
    Get sample content from a file for LLM context.
    
    Args:
        file_path: Path to the file
        max_lines: Maximum number of lines to return
        
    Returns:
        Sample content as string
    """
    try:
        if file_path.suffix.lower() == '.csv':
            df = pd.read_csv(file_path, nrows=max_lines)
            return df.to_string()
        elif file_path.suffix.lower() == '.json':
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return json.dumps(data, indent=2)[:1000]  # Limit to 1000 chars
        else:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = []
                for i, line in enumerate(f):
                    if i >= max_lines:
                        break
                    lines.append(line.rstrip())
                return '\n'.join(lines)
    except Exception:
        return f"Could not read sample from {file_path.name}"

def estimate_execution_time(analysis_type: str, file_sizes: List[int]) -> int:
    """
    Estimate execution time based on analysis type and file sizes.
    
    Args:
        analysis_type: Type of analysis
        file_sizes: List of file sizes in bytes
        
    Returns:
        Estimated timeout in seconds
    """
    base_timeout = 30
    total_size_mb = sum(file_sizes) / (1024 * 1024)
    
    # Adjust timeout based on analysis type
    multipliers = {
        'network': 2.0,
        'statistical': 1.5,
        'timeseries': 1.8,
        'ml': 3.0,
        'general': 1.0
    }
    
    multiplier = multipliers.get(analysis_type, 1.0)
    
    # Add time based on file size
    size_timeout = min(total_size_mb * 5, 120)  # Max 2 minutes for size
    
    return int(base_timeout * multiplier + size_timeout)

def validate_generated_code(code: str) -> Tuple[bool, str]:
    """
    Perform basic validation on generated code.
    
    Args:
        code: Python code to validate
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    # Check for dangerous operations - but be more permissive for data analysis
    dangerous_patterns = [
        'import subprocess',
        'eval(',
        'exec(',
        '__import__',
        'input(',
        'raw_input(',
        'execfile(',
        'reload(',
        'exit(',
        'quit(',
        'import socket',
        'import urllib',
        'import http',
        'import shutil',
        'rmdir(',
        'rm(',
        'delete(',
        'os.system',
        'os.popen',
        'os.remove',
        'os.rmdir'
    ]
    
    # Allow specific safe imports and operations
    safe_patterns = [
        'import pandas',
        'import numpy',
        'import matplotlib',
        'import networkx',
        'import json',
        'import csv',
        'import base64',
        'import requests',
        'from requests',
        'from bs4',
        'import beautifulsoup4',
        'from beautifulsoup4',
        'from pandas',
        'from numpy',
        'from matplotlib',
        'from networkx',
        'import seaborn',
        'from seaborn',
        'import plotly',
        'from plotly',
        'import scipy',
        'from scipy',
        'from pathlib',
        'from io',
        'import pathlib',
        'import io',
        'open(',  # Allow file operations
        'with open(',  # Allow file operations
        '.read(',
        '.write(',
        '.to_csv(',
        '.to_json(',
        '.savefig(',
        'json.dump',
        'json.load'
    ]
    
    lines = code.split('\n')
    for i, line in enumerate(lines, 1):
        line_lower = line.lower().strip()
        
        # Skip empty lines and comments
        if not line_lower or line_lower.startswith('#'):
            continue
        
        # Check for dangerous patterns
        for pattern in dangerous_patterns:
            if pattern in line_lower:
                return False, f"Potentially dangerous operation on line {i}: {pattern}"
    
    # Basic syntax check
    try:
        compile(code, '<string>', 'exec')
    except SyntaxError as e:
        return False, f"Syntax error: {str(e)}"
    
    return True, "Code validation passed"

def cleanup_sandbox_directory(sandbox_path: Path) -> None:
    """
    Clean up sandbox directory after execution.
    
    Args:
        sandbox_path: Path to sandbox directory to clean up
    """
    try:
        if sandbox_path.exists() and sandbox_path.is_dir():
            shutil.rmtree(sandbox_path)
    except Exception:
        pass  # Ignore cleanup errors

def format_file_size(size_bytes: int) -> str:
    """
    Format file size in human-readable format.
    
    Args:
        size_bytes: Size in bytes
        
    Returns:
        Formatted size string
    """
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f} TB"

def infer_task_type(question_text: str) -> str:
    """
    Infer the task type based on keywords in the question text.
    
    Args:
        question_text: The analysis question text
        
    Returns:
        Task type: "graph", "statistical", "timeseries", "scrape", or "custom"
    """
    if not question_text:
        return "custom"
    
    # Convert to lowercase for case-insensitive matching
    text_lower = question_text.lower()
    
    # Define keyword patterns for each task type
    graph_keywords = [
        "shortest path", "degree", "density", "centrality", "clustering",
        "network", "graph", "node", "edge", "connectivity", "community",
        "betweenness", "eigenvector", "pagerank", "diameter", "clique"
    ]
    
    statistical_keywords = [
        "average", "correlation", "median", "mean", "std", "variance",
        "distribution", "histogram", "statistics", "outlier", "percentile",
        "regression", "anova", "t-test", "chi-square", "p-value",
        "hypothesis", "confidence", "significance", "normal", "skewness"
    ]
    
    timeseries_keywords = [
        "time", "cumulative", "date", "trend", "seasonal", "forecast",
        "moving average", "rolling", "lag", "autocorrelation", "arima",
        "timestamp", "temporal", "chronological", "period", "frequency",
        "decomposition", "stationarity", "differencing"
    ]
    
    scrape_keywords = [
        "scrape", "website", "html", "web", "crawl", "extract", "parse",
        "beautifulsoup", "requests", "url", "http", "tag", "element",
        "xpath", "css selector", "dom"
    ]
    
    # Count keyword matches for each category
    scores = {
        "graph": sum(1 for keyword in graph_keywords if keyword in text_lower),
        "statistical": sum(1 for keyword in statistical_keywords if keyword in text_lower),
        "timeseries": sum(1 for keyword in timeseries_keywords if keyword in text_lower),
        "scrape": sum(1 for keyword in scrape_keywords if keyword in text_lower)
    }
    
    # Return the category with the highest score, or "custom" if no clear match
    if max(scores.values()) == 0:
        return "custom"
    
    return max(scores, key=scores.get)

def preview_file(file_path: Path, max_lines: int = 10) -> Dict[str, Any]:
    """
    Get a comprehensive preview of a file including metadata and content sample.
    
    Args:
        file_path: Path to the file to preview
        max_lines: Maximum number of lines to preview
        
    Returns:
        Dictionary containing file preview information
    """
    if not file_path.exists():
        return {
            "error": f"File {file_path} does not exist",
            "exists": False
        }
    
    preview = {
        "filename": file_path.name,
        "path": str(file_path),
        "size_bytes": file_path.stat().st_size,
        "size_formatted": format_file_size(file_path.stat().st_size),
        "extension": file_path.suffix.lower(),
        "exists": True,
        "content_preview": "",
        "metadata": {}
    }
    
    try:
        # File type specific preview
        if file_path.suffix.lower() == '.csv':
            preview.update(_preview_csv(file_path, max_lines))
        elif file_path.suffix.lower() == '.json':
            preview.update(_preview_json(file_path))
        elif file_path.suffix.lower() in ['.xlsx', '.xls']:
            preview.update(_preview_excel(file_path, max_lines))
        elif file_path.suffix.lower() in ['.html', '.htm']:
            preview.update(_preview_html(file_path, max_lines))
        else:
            preview.update(_preview_text(file_path, max_lines))
            
    except Exception as e:
        preview["error"] = f"Error reading file: {str(e)}"
    
    return preview

def _preview_csv(file_path: Path, max_lines: int) -> Dict[str, Any]:
    """Preview CSV file content."""
    try:
        # Read sample of CSV file
        df = pd.read_csv(file_path, nrows=max_lines)
        
        return {
            "file_type": "csv",
            "metadata": {
                "columns": list(df.columns),
                "num_columns": len(df.columns),
                "sample_rows": len(df),
                "dtypes": df.dtypes.astype(str).to_dict()
            },
            "content_preview": df.to_string(max_rows=max_lines, max_cols=10)
        }
    except Exception as e:
        # Fallback to raw text preview
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            lines = [f.readline().strip() for _ in range(max_lines)]
            return {
                "file_type": "csv",
                "content_preview": '\n'.join(lines),
                "error": f"CSV parse error: {str(e)}"
            }

def _preview_json(file_path: Path) -> Dict[str, Any]:
    """Preview JSON file content."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        metadata = {
            "data_type": type(data).__name__
        }
        
        if isinstance(data, list):
            metadata["length"] = len(data)
            if data and isinstance(data[0], dict):
                metadata["sample_keys"] = list(data[0].keys())
        elif isinstance(data, dict):
            metadata["keys"] = list(data.keys())
        
        # Create preview (limit size)
        preview_text = json.dumps(data, indent=2)
        if len(preview_text) > 1000:
            preview_text = preview_text[:1000] + "\n... (truncated)"
        
        return {
            "file_type": "json",
            "metadata": metadata,
            "content_preview": preview_text
        }
    except Exception as e:
        return {
            "file_type": "json",
            "error": f"JSON parse error: {str(e)}"
        }

def _preview_excel(file_path: Path, max_lines: int) -> Dict[str, Any]:
    """Preview Excel file content."""
    try:
        excel_file = pd.ExcelFile(file_path)
        sheets = excel_file.sheet_names
        
        metadata = {
            "sheets": sheets,
            "num_sheets": len(sheets)
        }
        
        # Preview first sheet
        if sheets:
            df = pd.read_excel(file_path, sheet_name=sheets[0], nrows=max_lines)
            metadata["first_sheet_columns"] = list(df.columns)
            preview_text = f"Sheet: {sheets[0]}\n" + df.to_string(max_rows=max_lines, max_cols=10)
        else:
            preview_text = "No sheets found"
        
        return {
            "file_type": "excel",
            "metadata": metadata,
            "content_preview": preview_text
        }
    except Exception as e:
        return {
            "file_type": "excel",
            "error": f"Excel parse error: {str(e)}"
        }

def _preview_html(file_path: Path, max_lines: int) -> Dict[str, Any]:
    """Preview HTML file content."""
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        
        # Basic HTML analysis
        metadata = {
            "size_chars": len(content),
            "has_tables": '<table' in content.lower(),
            "has_forms": '<form' in content.lower(),
            "has_scripts": '<script' in content.lower(),
            "title": ""
        }
        
        # Extract title if present
        title_match = re.search(r'<title[^>]*>([^<]+)</title>', content, re.IGNORECASE)
        if title_match:
            metadata["title"] = title_match.group(1).strip()
        
        # Create preview (first few lines)
        lines = content.split('\n')
        preview_lines = lines[:max_lines]
        preview_text = '\n'.join(preview_lines)
        
        if len(preview_text) > 1000:
            preview_text = preview_text[:1000] + "\n... (truncated)"
        
        return {
            "file_type": "html",
            "metadata": metadata,
            "content_preview": preview_text
        }
    except Exception as e:
        return {
            "file_type": "html",
            "error": f"HTML parse error: {str(e)}"
        }

def _preview_text(file_path: Path, max_lines: int) -> Dict[str, Any]:
    """Preview text file content."""
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            lines = []
            char_count = 0
            line_count = 0
            
            for line in f:
                if line_count >= max_lines:
                    break
                lines.append(line.rstrip())
                char_count += len(line)
                line_count += 1
        
        metadata = {
            "preview_lines": len(lines),
            "preview_chars": char_count,
            "encoding": "utf-8"
        }
        
        return {
            "file_type": "text",
            "metadata": metadata,
            "content_preview": '\n'.join(lines)
        }
    except Exception as e:
        return {
            "file_type": "text",
            "error": f"Text parse error: {str(e)}"
        }

def pre_scrape_data(question_text: str, sandbox_path: Path) -> list[Path]:
    """
    Pre-scrape data from URLs found in question_text and save into sandbox.
    Uses HTML cleaning for better data extraction.
    Returns list of file paths of saved scraped data.
    """
    from .html_cleaner import (
        scrape_and_clean_url, extract_table_metadata, 
        create_table_extraction_guide, create_wikipedia_film_extraction_guide
    )
    from .logger import logger
    
    scraped_files: list[Path] = []
    urls = re.findall(r'https?://[^\s]+', question_text)
    
    if not urls:
        logger.info("No URLs found in question text")
        return scraped_files
    
    # Check if this is a SQL/DuckDB query that should use database connections instead of web scraping
    if detect_sql_database_query(question_text):
        logger.info("Detected SQL/DuckDB query with database URLs - skipping web scraping")
        logger.info("URLs will be handled as database connections in the generated code")
        return scraped_files
    
    # Extract keywords from question for targeted cleaning
    target_keywords = extract_keywords_from_question(question_text)
    
    logger.info(f"Pre-scraping found {len(urls)} URLs: {urls}")
    logger.info(f"Target keywords: {target_keywords}")
    
    for url in urls:
        try:
            logger.info(f"Attempting to scrape and clean URL: {url}")
            
            # Use HTML cleaner for better data extraction
            cleaned_html, metadata, summary = scrape_and_clean_url(
                url=url, 
                target_keywords=target_keywords,
                save_path=sandbox_path / 'scraped_data_cleaned.html'
            )
            
            logger.info(f"Successfully scraped and cleaned URL: {url}")
            logger.info(f"Metadata: {metadata}")
            
            # Save HTML structure summary for LLM reference
            summary_path = sandbox_path / 'html_structure_summary.txt'
            with open(summary_path, 'w', encoding='utf-8') as f:
                f.write(summary)
            scraped_files.append(summary_path)
            logger.info(f"Saved HTML structure summary to: {summary_path}")
            
            # Create specialized extraction guide for Wikipedia film data
            if 'wikipedia.org' in url and any(keyword in question_text.lower() for keyword in ['film', 'movie', 'gross', 'box office']):
                film_guide_path = sandbox_path / 'wikipedia_film_extraction_guide.txt'
                film_guide = create_wikipedia_film_extraction_guide(cleaned_html)
                with open(film_guide_path, 'w', encoding='utf-8') as f:
                    f.write(film_guide)
                scraped_files.append(film_guide_path)
                logger.info(f"Created specialized Wikipedia film extraction guide: {film_guide_path}")
            else:
                # Create general table extraction guide
                table_guide_path = sandbox_path / 'table_extraction_guide.txt'
                table_guide = create_table_extraction_guide(cleaned_html)
                with open(table_guide_path, 'w', encoding='utf-8') as f:
                    f.write(table_guide)
                scraped_files.append(table_guide_path)
                logger.info(f"Created table extraction guide: {table_guide_path}")
            
            # Attempt to parse tabular data from cleaned HTML
            try:
                import pandas as pd
                from io import StringIO
                # Try multiple table extraction strategies
                tables = pd.read_html(StringIO(cleaned_html), header=0)
                
                if tables:
                    # Find the best table for film data
                    best_table = None
                    best_score = 0
                    
                    for i, table in enumerate(tables):
                        score = 0
                        # Score based on table characteristics
                        if table.shape[0] > 20:  # Good number of rows
                            score += 10
                        if table.shape[1] >= 4:  # Multiple columns
                            score += 5
                            
                        # Check column names for film-related content
                        col_text = ' '.join([str(col).lower() for col in table.columns])
                        if any(keyword in col_text for keyword in ['rank', 'title', 'gross', 'film']):
                            score += 15
                            
                        # Check table content for film indicators
                        table_text = table.to_string().lower()
                        film_indicators = sum(1 for word in ['avatar', 'avengers', 'titanic', 'billion'] if word in table_text)
                        score += film_indicators * 2
                        
                        if score > best_score:
                            best_score = score
                            best_table = table
                            logger.info(f"Table {i} scored {score} points - new best candidate")
                    
                    if best_table is not None:
                        # Clean the best table
                        main_table = best_table.copy()
                        
                        # Clean column names
                        if main_table.shape[1] >= 5:
                            # Common Wikipedia film table structure
                            new_columns = ['Rank', 'Peak', 'Title', 'Worldwide_Gross', 'Year']
                            if main_table.shape[1] > 5:
                                new_columns.extend([f'Column_{i+6}' for i in range(main_table.shape[1] - 5)])
                            main_table.columns = new_columns[:main_table.shape[1]]
                        
                        # Remove rows that are clearly headers or separators
                        main_table = main_table[main_table['Rank'].astype(str).str.match(r'^\d+$', na=False)]
                        
                        # Clean numeric columns
                        if 'Worldwide_Gross' in main_table.columns:
                            main_table['Worldwide_Gross'] = main_table['Worldwide_Gross'].astype(str)
                            main_table['Worldwide_Gross'] = main_table['Worldwide_Gross'].str.replace('$', '', regex=False)
                            main_table['Worldwide_Gross'] = main_table['Worldwide_Gross'].str.replace(',', '', regex=False)
                        
                        csv_path = sandbox_path / 'scraped_data.csv'
                        main_table.to_csv(csv_path, index=False)
                        scraped_files.append(csv_path)
                        logger.info(f"Extracted and cleaned tabular data to CSV: {csv_path} (shape: {main_table.shape})")
                        
                        # Save extraction metadata
                        extraction_metadata = {
                            'url': url,
                            'tables_found': len(tables),
                            'main_table_shape': main_table.shape,
                            'columns': list(main_table.columns),
                            'best_table_score': best_score,
                            'sample_data': main_table.head(3).to_dict('records') if len(main_table) > 0 else [],
                            'cleaning_metadata': metadata
                        }
                        metadata_path = sandbox_path / 'extraction_metadata.json'
                        with open(metadata_path, 'w', encoding='utf-8') as f:
                            import json
                            json.dump(extraction_metadata, f, indent=2, default=str)
                        scraped_files.append(metadata_path)
                        logger.info(f"Saved extraction metadata: {metadata_path}")
                    else:
                        logger.warning("No suitable table found among extracted tables")
                        
            except Exception as table_error:
                logger.warning(f"Could not extract tabular data from cleaned HTML: {table_error}")
                # Still save the cleaned HTML for manual extraction
                pass
                
        except Exception as e:
            logger.error(f"Error scraping URL {url}: {e}")
            # Try to save raw HTML as fallback
            try:
                import requests
                headers = {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
                }
                response = requests.get(url, headers=headers, timeout=30)
                response.raise_for_status()
                
                raw_html_path = sandbox_path / 'scraped_data.html'
                with open(raw_html_path, 'w', encoding='utf-8') as f:
                    f.write(response.text)
                scraped_files.append(raw_html_path)
                logger.info(f"Saved raw HTML as fallback: {raw_html_path}")
                
            except Exception as fallback_error:
                logger.error(f"Failed to save even raw HTML for {url}: {fallback_error}")
                pass
    
    logger.info(f"Pre-scraping completed. Created {len(scraped_files)} files: {[f.name for f in scraped_files]}")
    return scraped_files


def extract_keywords_from_question(question_text: str) -> List[str]:
    """
    Extract relevant keywords from question text for targeted HTML cleaning.
    
    Args:
        question_text: Natural language question
        
    Returns:
        List of keywords for HTML cleaning
    """
    # Common data-related keywords
    base_keywords = ['table', 'data', 'list', 'ranking', 'information']
    
    # Extract domain-specific keywords
    question_lower = question_text.lower()
    
    # Movie/Entertainment keywords
    if any(word in question_lower for word in ['movie', 'film', 'box office', 'gross', 'cinema']):
        base_keywords.extend(['highest-grossing', 'box office', 'worldwide', 'gross', 'revenue', 'film', 'movie'])
    
    # Business/Sales keywords
    if any(word in question_lower for word in ['sales', 'revenue', 'business', 'profit', 'financial']):
        base_keywords.extend(['sales', 'revenue', 'profit', 'business', 'financial', 'earnings'])
    
    # Sports keywords
    if any(word in question_lower for word in ['sport', 'team', 'player', 'score', 'match']):
        base_keywords.extend(['team', 'player', 'score', 'match', 'season', 'league'])
    
    # Financial/Stock keywords
    if any(word in question_lower for word in ['stock', 'share', 'market', 'price', 'trading']):
        base_keywords.extend(['stock', 'share', 'price', 'market', 'trading', 'index'])
    
    # Geographic/Country keywords
    if any(word in question_lower for word in ['country', 'region', 'city', 'population', 'geographic']):
        base_keywords.extend(['country', 'region', 'population', 'area', 'capital'])
    
    # Extract explicit nouns and important terms
    import re
    words = re.findall(r'\b[a-zA-Z]+\b', question_text)
    important_words = [word for word in words if len(word) > 4 and word.lower() not in 
                      ['this', 'that', 'with', 'from', 'have', 'been', 'will', 'what', 'where', 'when']]
    
    base_keywords.extend(important_words[:5])  # Add up to 5 important words
    
    return list(set(base_keywords))  # Remove duplicates

def detect_sql_database_query(question_text: str) -> bool:
    """
    Detect if the question contains SQL/DuckDB queries with URLs that should be handled
    as database connections rather than web scraping.
    
    Args:
        question_text: The question text to analyze
        
    Returns:
        True if SQL/DuckDB query detected, False otherwise
    """
    # Convert to lowercase for case-insensitive matching
    text_lower = question_text.lower()
    
    # SQL/DuckDB specific patterns
    sql_patterns = [
        'select',
        'from',
        'where',
        'group by',
        'order by',
        'install',
        'load',
        'duckdb',
        'read_parquet',
        'read_csv',
        'read_json',
        'create table',
        'insert into',
        'update',
        'delete from'
    ]
    
    # Database connection patterns  
    db_patterns = [
        's3://',
        'read_parquet(',
        'read_csv(',
        'read_json(',
        '.parquet',
        '.csv',
        '.json',
        'httpfs',
        'install httpfs',
        'load httpfs',
        's3_region=',
        'bucket',
        'aws',
        'gcs://',
        'gs://'
    ]
    
    # URL patterns that are likely database sources (not web pages to scrape)
    database_url_patterns = [
        's3://[^\\s]+\\.parquet',
        's3://[^\\s]+\\.csv', 
        's3://[^\\s]+\\.json',
        'gs://[^\\s]+\\.parquet',
        'gs://[^\\s]+\\.csv',
        'gs://[^\\s]+\\.json',
        'https?://[^\\s]+\\.parquet',
        'https?://[^\\s]+\\.csv',
        'https?://[^\\s]+\\.json'
    ]
    
    # Check for SQL keywords
    sql_keyword_count = sum(1 for pattern in sql_patterns if pattern in text_lower)
    
    # Check for database patterns
    db_pattern_count = sum(1 for pattern in db_patterns if pattern in text_lower)
    
    # Check for database URL patterns using regex
    import re
    database_url_matches = any(re.search(pattern, question_text, re.IGNORECASE) for pattern in database_url_patterns)
    
    # Decision logic:
    # - If we have multiple SQL keywords AND database patterns, it's likely a SQL query
    # - If we have database URL patterns, it's likely a data source rather than web scraping
    is_sql_query = (
        (sql_keyword_count >= 2 and db_pattern_count >= 1) or  # Multiple SQL keywords + DB patterns
        database_url_matches or  # Direct database file URLs
        ('install httpfs' in text_lower and 'load httpfs' in text_lower) or  # DuckDB setup
        ('read_parquet' in text_lower and 's3://' in text_lower)  # Direct DuckDB + S3
    )
    
    return is_sql_query

def auto_detect_analysis_type(file_paths: List[Path], question: str = "") -> str:
    """
    Automatically detect the most appropriate analysis type based on file content and question.
    
    Args:
        file_paths: List of file paths to analyze
        question: User's question (optional for additional context)
        
    Returns:
        Suggested analysis type
    """
    question_lower = question.lower()
    
    # Check for explicit analysis type hints in the question
    if any(word in question_lower for word in ['network', 'graph', 'node', 'edge', 'connection']):
        return 'network'
    elif any(word in question_lower for word in ['time', 'trend', 'seasonal', 'forecast', 'over time']):
        return 'timeseries'
    elif any(word in question_lower for word in ['cluster', 'classify', 'predict', 'model', 'machine learning']):
        return 'ml'
    elif any(word in question_lower for word in ['sql', 'query', 'database', 'select', 'from', 'where']):
        return 'database'
    elif any(word in question_lower for word in ['statistical', 'correlation', 'distribution', 'mean', 'median']):
        return 'statistical'
    
    # Analyze file content for hints
    json_files = [f for f in file_paths if f.suffix.lower() == '.json']
    
    if json_files:
        # Sample first JSON file for structure analysis
        try:
            file_analysis = analyze_json_file(json_files[0])
            structure = file_analysis.get('structure', {})
            
            # Use the suggested analysis from structure analysis
            suggested = structure.get('suggested_analysis', 'json')
            if suggested != 'general':
                return suggested
                
            # Additional heuristics
            keys = structure.get('keys', [])
            if keys:
                key_text = ' '.join(keys).lower()
                
                # Time series indicators
                if any(time_word in key_text for time_word in ['date', 'time', 'year', 'month', 'timestamp', 'created', 'updated']):
                    return 'timeseries'
                
                # Network indicators  
                if any(net_word in key_text for net_word in ['source', 'target', 'from', 'to', 'node', 'edge', 'connection']):
                    return 'network'
                
                # Statistical indicators
                if any(stat_word in key_text for stat_word in ['value', 'score', 'rating', 'price', 'amount', 'quantity']):
                    return 'statistical'
            
            # Complex nested structures suggest JSON-specific analysis
            if structure.get('complexity') in ['complex', 'nested']:
                return 'json'
                
        except Exception:
            pass
    
    # Default fallback
    return 'general'