"""
LLM integration module for generating analysis code using Google's Gemini API.
"""
import os
import json
import asyncio
import re
from pathlib import Path
from typing import List, Dict, Any, Optional
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold

from .logger import setup_logger, log_llm_interaction, logger
from .utils import analyze_file_structure
from dotenv import load_dotenv
# Load environment variables
load_dotenv()
# Initialize logger (use global logger from logger module)
# logger = setup_logger()

# Configure Gemini API
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

# Model configuration
MODEL_NAME = "gemini-1.5-flash"  # Using the free tier model
GENERATION_CONFIG = {
    "temperature": 0.1,
    "top_p": 0.95,
    "top_k": 64,
    "max_output_tokens": 8192,
}

# Safety settings - allow code generation
SAFETY_SETTINGS = {
    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
}

async def generate_analysis_code(
    question: str,
    file_paths: List[Path],
    analysis_type: str,
    sandbox_path: Path,
    request_id: str = "unknown"
) -> Optional[str]:
    """
    Generate Python analysis code using Gemini API.
    
    Args:
        question: Natural language analysis question
        file_paths: List of file paths to analyze
        analysis_type: Type of analysis requested
        sandbox_path: Path to sandbox directory
        request_id: Request ID for logging
        
    Returns:
        Generated Python code or None if generation fails
    """
    if not GEMINI_API_KEY:
        logger.error("GEMINI_API_KEY environment variable not set")
        return None
    
    try:
        # Analyze file structure
        file_analysis = analyze_file_structure(file_paths)
        
        # Create prompt
        prompt = create_analysis_prompt(
            question=question,
            file_analysis=file_analysis,
            file_paths=file_paths,
            analysis_type=analysis_type,
            sandbox_path=sandbox_path
        )
        
        # Initialize model
        model = genai.GenerativeModel(
            model_name=MODEL_NAME,
            generation_config=GENERATION_CONFIG,
            safety_settings=SAFETY_SETTINGS
        )
        
        # Generate response
        response = await asyncio.to_thread(model.generate_content, prompt)
        
        if not response or not response.text:
            logger.error(f"Empty response from Gemini API for request {request_id}")
            return None
        
        # Log interaction
        log_llm_interaction(
            logger=logger,
            request_id=request_id,
            prompt_length=len(prompt),
            response_length=len(response.text),
            model_used=MODEL_NAME
        )
        
        # Extract code from response
        generated_code = extract_code_from_response(response.text)
        
        if not generated_code:
            logger.error(f"Could not extract code from Gemini response for request {request_id}")
            return None
        
        # Validate generated code
        from .utils import validate_generated_code
        is_valid, error_msg = validate_generated_code(generated_code)
        
        if not is_valid:
            logger.error(f"Generated code validation failed for request {request_id}: {error_msg}")
            return None
        
        logger.info(f"Successfully generated analysis code for request {request_id}")
        return generated_code
        
    except Exception as e:
        logger.error(f"Error generating analysis code for request {request_id}: {str(e)}")
        return None

def extract_output_structure_from_question(question: str) -> Dict[str, Any]:
    """
    Extract the required output structure from the question text with enhanced detection.
    
    Args:
        question: User's question text
        
    Returns:
        Dictionary containing output format information
    """
    output_info = {
        "format_type": "array",  # "object" or "array"
        "structure": None,
        "required_keys": [],
        "array_elements": [],
        "strict_compliance": True
    }
    
    # Enhanced JSON object format detection
    object_patterns = [
        "Return a JSON object with keys:",
        "JSON object with keys:",
        "return a JSON object with keys:",
        "json object with keys:",
        "Response format: JSON object with keys:",
        "Output format: JSON object with keys:"
    ]
    
    if any(pattern in question for pattern in object_patterns):
        output_info["format_type"] = "object"
        
        # Extract keys and their types with enhanced parsing
        lines = question.split('\n')
        in_keys_section = False
        for line in lines:
            # Start detection
            if any(pattern.lower() in line.lower() for pattern in ["with keys:", "keys:"]):
                in_keys_section = True
                continue
            
            # Key parsing
            if in_keys_section and line.strip().startswith('- `'):
                # Parse key definition like "- `edge_count`: number"
                key_def = line.strip()[3:]  # Remove "- `"
                if '`:' in key_def:
                    key_name = key_def.split('`:')[0]
                    key_type = key_def.split('`: ')[1] if '`: ' in key_def else "string"
                    output_info["required_keys"].append({"name": key_name, "type": key_type})
            elif in_keys_section and line.strip().startswith('- '):
                # Alternative format without backticks
                key_def = line.strip()[2:]  # Remove "- "
                if ':' in key_def:
                    parts = key_def.split(':', 1)
                    key_name = parts[0].strip().strip('`')
                    key_type = parts[1].strip() if len(parts) > 1 else "string"
                    output_info["required_keys"].append({"name": key_name, "type": key_type})
            elif in_keys_section and line.strip() and not line.strip().startswith('-') and not line.strip().startswith('Answer'):
                # End of keys section (unless it's the Answer: section)
                break
    
    # Enhanced JSON array format detection
    array_patterns = [
        "JSON array of strings",
        "Respond with a JSON array",
        "return a JSON array",
        "json array",
        "Response format: JSON array",
        "Output format: JSON array",
        "array of strings containing",
        "JSON array containing"
    ]
    
    if any(pattern.lower() in question.lower() for pattern in array_patterns):
        output_info["format_type"] = "array"
        
        # Count numbered questions to determine array size with better regex
        import re
        numbered_questions = re.findall(r'^\s*\d+\.\s', question, re.MULTILINE)
        output_info["array_elements"] = len(numbered_questions)
        
        # Also check for explicit size mentions
        size_patterns = [
            r'(\d+)\s*(?:element|item|string)s?\s*(?:in\s*(?:the\s*)?array|array)',
            r'array\s*(?:of|with)\s*(\d+)\s*(?:element|item|string)s?',
            r'(\d+)-element\s*array'
        ]
        
        for pattern in size_patterns:
            match = re.search(pattern, question.lower())
            if match:
                output_info["array_elements"] = int(match.group(1))
                break
    
    # Fallback detection for legacy format mentions
    if not output_info["required_keys"] and output_info["array_elements"] == 0:
        # Look for explicit element count mentions
        import re
        
        # Check for "4 element array" or similar patterns
        element_patterns = [
            r'(\d+)\s*element\s*array',
            r'(\d+)\s*item\s*array', 
            r'array\s*of\s*(\d+)\s*elements?',
            r'exactly\s*(\d+)\s*elements?',
            r'(\d+)\s*answers?'
        ]
        
        for pattern in element_patterns:
            match = re.search(pattern, question.lower())
            if match:
                output_info["array_elements"] = int(match.group(1))
                break
        
        # Default fallback if still nothing detected
        if output_info["array_elements"] == 0:
            # Look for numbered questions as final attempt
            numbered_questions = re.findall(r'^\s*\d+\.\s', question, re.MULTILINE)
            if numbered_questions:
                output_info["array_elements"] = len(numbered_questions)
            else:
                # No fallback allowed - format must be explicit
                output_info["format_type"] = "unknown"
                output_info["array_elements"] = 0
    
    return output_info

def generate_output_format_instructions(output_structure: Dict[str, Any], question: str) -> str:
    """
    Generate specific instructions for the required output format with strict compliance.
    
    Args:
        output_structure: Output structure information extracted from question
        question: Original question text
        
    Returns:
        Formatted instructions for code generation
    """
    if output_structure["format_type"] == "object":
        # JSON object format - strict key compliance
        keys_info = []
        key_assignments = []
        
        for key_info in output_structure["required_keys"]:
            key_name = key_info["name"]
            key_type = key_info["type"]
            python_type = get_python_type(key_type)
            default_val = get_default_value(key_type)
            
            keys_info.append(f'  "{key_name}": {key_type}')
            key_assignments.append(f'result["{key_name}"] = safe_convert(your_calculated_{key_name.lower()}, {python_type}, {default_val})')
        
        keys_structure = "{\n" + ",\n".join(keys_info) + "\n}"
        key_implementation = "\n".join(key_assignments)
        
        return f"""
**🔒 STRICT OUTPUT FORMAT COMPLIANCE - JSON OBJECT:**
The question explicitly specifies a JSON object with these EXACT keys:
{keys_structure}

**❌ CRITICAL REQUIREMENTS - ZERO TOLERANCE FOR DEVIATION:**
1. ✅ MUST use exactly these {len(output_structure["required_keys"])} keys: {[k["name"] for k in output_structure["required_keys"]]}
2. ✅ MUST NOT add any additional keys
3. ✅ MUST NOT omit any required keys  
4. ✅ MUST use safe_convert() for all values
5. ✅ MUST save as JSON object, NOT array
6. ✅ For base64 images, include ONLY the base64 string (no prefix)

**📋 MANDATORY IMPLEMENTATION TEMPLATE:**
```python
# Initialize result dictionary with EXACT keys from question
result = {{}}

# Calculate and assign each required key (replace calculations with actual logic)
{key_implementation}

# MANDATORY: Validate all keys are present
required_keys = {[k["name"] for k in output_structure["required_keys"]]}
for key in required_keys:
    if key not in result:
        logger.error(f"Missing required key: {{key}}")
        result[key] = safe_convert(None, str, "Missing data")

# Save result as JSON OBJECT (not array)
with open('result.json', 'w') as f:
    json.dump(result, f, cls=RobustJSONEncoder, ensure_ascii=False, indent=2)
    
print(f"✅ Created JSON object with {{len(result)}} keys: {{list(result.keys())}}")
```

**🚨 VALIDATION CHECKLIST - MUST VERIFY BEFORE SAVING:**
□ Result is a dictionary (dict), not a list
□ Contains exactly {len(output_structure["required_keys"])} keys
□ All key names match exactly: {[k["name"] for k in output_structure["required_keys"]]}
□ No extra keys added
□ All values are JSON-serializable
□ Base64 images have proper data URI prefix
"""
    
    elif output_structure["format_type"] == "array":
        # JSON array format - strict element compliance
        num_elements = output_structure.get("array_elements", 4)
        
        return f"""
**🔒 STRICT OUTPUT FORMAT COMPLIANCE - JSON ARRAY OF STRINGS:**
The question explicitly requires a JSON array with EXACTLY {num_elements} string elements.
Each element must answer the corresponding numbered question (1, 2, 3, ..., {num_elements}).

**❌ CRITICAL REQUIREMENTS - ZERO TOLERANCE FOR DEVIATION:**
1. ✅ MUST return exactly {num_elements} elements (not {num_elements-1}, not {num_elements+1})
2. ✅ MUST be a JSON array (list), NOT a JSON object (dict)
3. ✅ ALL elements MUST be strings (use str() conversion)
4. ✅ Each element answers the corresponding numbered question
5. ✅ For base64 images, include ONLY the base64 string (no prefix)
6. ✅ Numeric answers must be converted to strings

**📋 MANDATORY IMPLEMENTATION TEMPLATE:**
```python
# Initialize results array with exact number of elements
results = []

# Answer each numbered question in order (replace with actual calculations)
# Question 1:
answer1 = safe_convert(your_calculation_1, str, "No data")
results.append(answer1)

# Question 2:
answer2 = safe_convert(your_calculation_2, str, "No data")
results.append(answer2)

{"".join([f'''
# Question {i+3}:
answer{i+3} = safe_convert(your_calculation_{i+3}, str, "No data")
results.append(answer{i+3})''' for i in range(max(0, num_elements-2))])}

# MANDATORY: Validate exact length and types
if len(results) != {num_elements}:
    logger.error(f"Wrong array length: {{len(results)}} instead of {num_elements}")
    # Fix length
    while len(results) < {num_elements}:
        results.append("No data")
    results = results[:{num_elements}]

# Ensure all elements are strings
results = [str(item) for item in results]

# Save result as JSON ARRAY (not object)
with open('result.json', 'w') as f:
    json.dump(results, f, cls=RobustJSONEncoder, ensure_ascii=False, indent=2)
    
print(f"✅ Created JSON array with {{len(results)}} string elements")
```

**🚨 VALIDATION CHECKLIST - MUST VERIFY BEFORE SAVING:**
□ Result is a list (array), not a dictionary (object)
□ Contains exactly {num_elements} elements
□ All elements are strings
□ Elements answer questions 1, 2, 3, ..., {num_elements} in order
□ Base64 images have proper data URI prefix
□ No nested arrays or objects
"""
    
    else:
        # 🚨 ERROR: No fallback formats allowed - question must specify exact structure
        return """
**❌ CRITICAL ERROR: UNABLE TO DETERMINE OUTPUT FORMAT**

The question does not clearly specify the required output format.

**🚨 MANDATORY FORMAT SPECIFICATIONS:**
The question MUST contain one of these clear indicators:
1. "Return a JSON object with keys:" followed by key specifications
2. "JSON array of strings" with numbered questions (1., 2., 3., etc.)

**📋 NO FALLBACK FORMATS ALLOWED:**
- No default 4-element arrays
- No sample data structures  
- No placeholder formats
- Must error if format unclear

**🔧 REQUIRED ACTION:**
The question must be updated to clearly specify the expected output structure.
No code generation is possible without explicit format specification.

**� IMPLEMENTATION:**
```python
# The LLM must raise an error if output format cannot be determined
raise ValueError("Question does not specify required output format. Must contain either 'JSON object with keys:' or 'JSON array of strings' specification.")
```
"""

def get_python_type(type_string: str) -> str:
    """Convert question type to Python type for safe_convert."""
    if "number" in type_string.lower() or "int" in type_string.lower():
        return "int"
    elif "float" in type_string.lower():
        return "float"
    elif "string" in type_string.lower():
        return "str"
    else:
        return "str"

def get_default_value(type_string: str) -> str:
    """Get default value for a type."""
    if "number" in type_string.lower() or "int" in type_string.lower():
        return "0"
    elif "float" in type_string.lower():
        return "0.0"
    elif "base64" in type_string.lower():
        return '""'  # Empty string for base64 images
    else:
        return '"No data"'

def create_analysis_prompt(
    question: str,
    file_analysis: Dict[str, Any],
    file_paths: List[Path],
    analysis_type: str,
    sandbox_path: Path
) -> str:
    """
    Create a detailed prompt for code generation.
    
    Args:
        question: User's analysis question
        file_analysis: Analysis of uploaded files
        file_paths: List of file paths
        analysis_type: Type of analysis
        sandbox_path: Path to sandbox directory
        
    Returns:
        Formatted prompt string
    """
    # Import here to avoid circular imports
    from .utils import get_all_available_files
    
    # Get all available files including any scraped data
    all_files = get_all_available_files(sandbox_path, file_paths)
    
    # Extract required output structure from question
    output_structure = extract_output_structure_from_question(question)
    
    # Check if scraped data files exist
    scraped_files = [f for f in all_files if 'scraped_data' in f.name]
    scraped_context = ""
    if scraped_files:
        scraped_context = f"""

**PREVIOUSLY SCRAPED DATA AVAILABLE:**
The following scraped data files are available from previous analysis steps:
{json.dumps([f.name for f in scraped_files], indent=2)}
You can use these files directly in your analysis instead of scraping again.
"""
    
    # Identify available data files for context (no sample content)
    data_files_info = []
    for file_path in all_files:
        if file_path.name != "question.txt":  # Exclude question file from data files
            file_info = {
                "name": file_path.name,
                "type": file_path.suffix.lower(),
                "size_bytes": file_path.stat().st_size if file_path.exists() else 0
            }
            data_files_info.append(file_info)
    
    # Generate output format instructions based on question structure
    output_format_instructions = generate_output_format_instructions(output_structure, question)
    
    prompt = f"""
🔒 **CRITICAL COMPLIANCE: NO SAMPLE DATA - WORKSPACE FILES ONLY**

You MUST analyze this question and determine the EXACT output structure required.
🚨 ALL DATA MUST COME FROM FILES IN THE SANDBOX WORKSPACE - NO SAMPLE/FALLBACK DATA ALLOWED.

**USER QUESTION:** {question}

🚨 **MANDATORY REQUIREMENTS:**
1. **OUTPUT FORMAT:** If question contains "Return a JSON object with keys:" → MUST return JSON object with exact keys
2. **OUTPUT FORMAT:** If question contains "JSON array of strings" → MUST return JSON array with string elements  
3. **OUTPUT FORMAT:** If question contains numbered questions (1., 2., 3.) → Array size = number of questions
4. **DATA SOURCE:** ALL data MUST be loaded from files present in the sandbox workspace
5. **JSON ATTACHMENTS:** JSON files (especially data.json, input.json) contain user-uploaded data
6. **NO FALLBACKS:** Zero tolerance for sample data, placeholder answers, or fallback values
7. **ERROR HANDLING:** If data missing or format unclear → RAISE ERROR (do not use sample data)

**ANALYSIS TYPE:** {analysis_type}

**WORKSPACE CONTEXT:**
📂 Available Files: {json.dumps(file_analysis, indent=2)}
📊 Data Files Detected: {json.dumps(data_files_info, indent=2)}{scraped_context}

🎯 **JSON ATTACHMENT INTELLIGENCE:**
- JSON files (*.json) often contain user attachments with structured data
- Common names: data.json, input.json, attachment.json, dataset.json
- Must handle various JSON structures: arrays of objects, nested dictionaries, key-value pairs
- Use pd.json_normalize() for complex structures, pd.DataFrame() for simple ones
- NO sample JSON data - only process real files from workspace

{output_format_instructions}

**🔒 MANDATORY JSON SERIALIZATION COMPLIANCE:**

**1. MANDATORY JSON Encoder Class:**
You MUST include and use this exact JSON encoder class to handle NumPy and Pandas data types:

```python
import json
import numpy as np
import pandas as pd

class RobustJSONEncoder(json.JSONEncoder):
    '''Handles all NumPy/Pandas data types that cause JSON serialization errors'''
    def default(self, obj):
        # Handle NumPy integers
        if isinstance(obj, np.integer):
            return int(obj)
        # Handle NumPy floats
        elif isinstance(obj, np.floating):
            return float(obj)
        # Handle NumPy arrays
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        # Handle NumPy bool
        elif isinstance(obj, np.bool_):
            return bool(obj)
        # Handle Pandas integers (Int64, int64)
        elif hasattr(obj, 'dtype') and 'int' in str(obj.dtype):
            return int(obj)
        # Handle Pandas floats
        elif hasattr(obj, 'dtype') and 'float' in str(obj.dtype):
            return float(obj)
        # Handle Pandas Series
        elif isinstance(obj, pd.Series):
            return obj.tolist()
        # Handle Pandas Timestamp
        elif isinstance(obj, pd.Timestamp):
            return obj.isoformat()
        # Handle datetime objects
        elif hasattr(obj, 'isoformat'):
            return obj.isoformat()
        # Handle NaN values
        elif pd.isna(obj):
            return None
        return super().default(obj)
```

**2. MANDATORY safe_convert Function:**
Always include this function to prevent type conversion errors:

```python
def safe_convert(value, target_type=float, default=0):
    '''Safely convert values to JSON-serializable types'''
    try:
        if pd.isna(value) or value is None:
            return default
        # Handle NumPy types
        if hasattr(value, 'item'):  # NumPy scalars have .item() method
            value = value.item()
        # Convert to target type
        if target_type == int:
            return int(float(value))  # Convert via float to handle strings
        elif target_type == float:
            return float(value)
        elif target_type == str:
            return str(value)
        else:
            return target_type(value)
    except (ValueError, TypeError, AttributeError):
        return default
```
        # Handle NumPy arrays
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        # Handle NumPy bool
        elif isinstance(obj, np.bool_):
            return bool(obj)
        # Handle Pandas integers (Int64, int64)
        elif hasattr(obj, 'dtype') and 'int' in str(obj.dtype):
            return int(obj)
        # Handle Pandas floats
        elif hasattr(obj, 'dtype') and 'float' in str(obj.dtype):
            return float(obj)
        # Handle Pandas Series
        elif isinstance(obj, pd.Series):
            return obj.tolist()
        # Handle Pandas Timestamp
        elif isinstance(obj, pd.Timestamp):
            return obj.isoformat()
        # Handle datetime objects
        elif hasattr(obj, 'isoformat'):
            return obj.isoformat()
        # Handle NaN values
        elif pd.isna(obj):
            return None
        return super().default(obj)
```

**2. MANDATORY safe_convert Function:**
Always include this function to prevent type conversion errors:

```python
def safe_convert(value, target_type=float, default=0):
    '''Safely convert values to JSON-serializable types'''
    try:
        if pd.isna(value) or value is None:
            return default
        # Handle NumPy types
        if hasattr(value, 'item'):  # NumPy scalars have .item() method
            value = value.item()
        # Convert to target type
        if target_type == int:
            return int(float(value))  # Convert via float to handle strings
        elif target_type == float:
            return float(value)
        elif target_type == str:
            return str(value)
        else:
            return target_type(value)
    except (ValueError, TypeError, AttributeError):
        return default
```

**3. MANDATORY Result Validation:**
Always validate results before JSON serialization:

```python
def validate_and_clean_results(results):
    '''Ensure all results are JSON serializable'''
    cleaned = []
    for i, item in enumerate(results):
        if i == 0:  # Numeric result
            cleaned.append(safe_convert(item, int, 0))
        elif i == 1:  # String result
            cleaned.append(safe_convert(item, str, "No data"))
        elif i == 2:  # Float result
            cleaned.append(safe_convert(item, float, 0.0))
        elif i == 3:  # Base64 image
            cleaned.append(str(item) if item else "")
        else:
            cleaned.append(safe_convert(item, str, ""))
    return cleaned
```

**4. Data Source Strategy:**
   - **SQL/DuckDB:** For questions with SQL queries, s3://, database URLs, .parquet/.csv remote files
   - **Web Scraping:** Only for explicit web content requests (Wikipedia pages, HTML tables)
   - **Local Files:** Default for uploaded files in current directory

**5. DuckDB Database Operations:**
   ```python
   import duckdb
   conn = duckdb.connect()
   # Install required extensions
   conn.execute("INSTALL httpfs; LOAD httpfs;")  # For remote URLs
   conn.execute("INSTALL parquet; LOAD parquet;")  # For parquet files
   
   # Query patterns:
   result = conn.execute("SELECT * FROM read_parquet('s3://bucket/file.parquet')").fetchdf()
   result = conn.execute("SELECT * FROM read_csv('https://example.com/data.csv')").fetchdf()
   conn.close()
   ```
   
   **DuckDB Function Reference:**
   - ✅ Date differences: `DATE_DIFF('day', date1, date2)` or `DATEDIFF('day', date1, date2)`
   - ✅ Date arithmetic: `date1 - date2` (for DATE columns)
   - ✅ Date casting: `column::DATE` or `CAST(column AS DATE)`
   - ✅ String aggregation: `STRING_AGG(column, ',')`
   - ❌ Never use SQLite functions: `JULIANDAY()`, `GROUP_CONCAT()`
   - ❌ Never use MySQL/PostgreSQL specific syntax

**6. Output Format (MANDATORY):**
   Save exactly 4 elements to 'result.json' using the RobustJSONEncoder:
   ```json
   [numeric_value, "string_value", float_value, "base64_encoded_image"]
   ```
   
   **Examples:**
   - ✅ CORRECT: `[42, "Product A", 0.85, "iVBORw0KGgoAAAANSUhEUgAAAo..."]`
   - ❌ WRONG: `["Count: 42", "Top product: Product A", "Correlation: 0.85"]`
   
   **Element Guidelines:**
   - [0]: Raw numeric answer (count, sum, ID, etc.)
   - [1]: String answer (name, category, description)
   - [2]: Calculated metric (average, correlation, percentage as decimal)
   - [3]: Visualization as base64 PNG string (just the base64 data, no prefix)

**7. Analysis Methodology:**
   ```python
   # 1. Data Discovery
   print("Data shape and types:")
   print(df.info())
   print("\nFirst few rows:")
   print(df.head())
   
   # 2. Data Quality Check
   print("\nMissing values:")
   print(df.isnull().sum())
   print("\nDuplicates:", df.duplicated().sum())
   
   # 3. Analysis Execution
   # Apply appropriate methods based on question
   
   # 4. Visualization Creation
   # Create relevant plots for the analysis
   
   # 5. Results Compilation with JSON Safety
   # Always use safe_convert and RobustJSONEncoder
   ```

**8. Error Handling & Robustness:**
   - Wrap file operations in try-except blocks
   - Validate data types and handle conversion errors
   - Check for empty datasets before analysis
   - Provide fallback values if calculations fail
   - Use `df.fillna()` or `df.dropna()` for missing data
   - Handle edge cases (single row, all same values, etc.)
   - Always use RobustJSONEncoder for JSON serialization

**OUTPUT FORMAT:**
Your response should contain only Python code between ```python and ``` markers.

🔒 **FINAL CRITICAL REMINDER - OUTPUT FORMAT COMPLIANCE:**

The code you generate MUST:
1. 🔍 **ANALYZE** the question to detect the exact output format required
2. ✅ **IMPLEMENT** the precise structure specified (JSON object with exact keys OR JSON array with exact element count)
3. 🚨 **VALIDATE** the result format before saving to ensure zero deviation
4. 📋 **FOLLOW** the output format instructions exactly as provided above

❌ **NEVER:**
- Change the required output structure
- Add extra keys to JSON objects  
- Use wrong array lengths
- Return arrays when objects are required (or vice versa)
- Ignore format specifications

✅ **ALWAYS:**
- Use RobustJSONEncoder for JSON serialization
- Use safe_convert() for all data type conversions
- Include validation checks before saving
- Print format compliance confirmations

🎯 **SUCCESS CRITERIA:** 
1. 📊 **DATA COMPLIANCE:** ALL data must come from files in the sandbox workspace (especially JSON attachments)
2. 🔒 **FORMAT COMPLIANCE:** Output must exactly match the structure specified in the question
3. 🚫 **NO SAMPLE DATA:** Zero tolerance for placeholder, sample, or fallback data
4. ✅ **JSON SAFETY:** Must use RobustJSONEncoder and safe_convert() for all serialization
5. 🎯 **WORKSPACE FOCUS:** JSON attachments (data.json, etc.) are primary data sources

🚨 **ABSOLUTE REQUIREMENTS:**
- Load data ONLY from workspace files (prioritize JSON attachments)  
- Follow question format specification exactly (object keys OR array length)
- Raise errors if data missing (never use sample/fallback data)
- Implement actual calculations from real data (no placeholder answers)

Generate the Python code now:

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import base64
from pathlib import Path
from io import BytesIO
import duckdb
import requests
from bs4 import BeautifulSoup

# Configure matplotlib
plt.switch_backend('Agg')
plt.style.use('default')

class RobustJSONEncoder(json.JSONEncoder):
    '''Handles all NumPy/Pandas data types that cause JSON serialization errors'''
    def default(self, obj):
        # Handle NumPy integers
        if isinstance(obj, np.integer):
            return int(obj)
        # Handle NumPy floats
        elif isinstance(obj, np.floating):
            return float(obj)
        # Handle NumPy arrays
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        # Handle NumPy bool
        elif isinstance(obj, np.bool_):
            return bool(obj)
        # Handle Pandas integers (Int64, int64)
        elif hasattr(obj, 'dtype') and 'int' in str(obj.dtype):
            return int(obj)
        # Handle Pandas floats
        elif hasattr(obj, 'dtype') and 'float' in str(obj.dtype):
            return float(obj)
        # Handle Pandas Series
        elif isinstance(obj, pd.Series):
            return obj.tolist()
        # Handle Pandas Timestamp
        elif isinstance(obj, pd.Timestamp):
            return obj.isoformat()
        # Handle datetime objects
        elif hasattr(obj, 'isoformat'):
            return obj.isoformat()
        # Handle NaN values
        elif pd.isna(obj):
            return None
        return super().default(obj)

def create_visualization(data, title="Analysis Results"):
    # Create appropriate visualization based on data type and analysis
    plt.figure(figsize=(10, 6))
    
    try:
        if hasattr(data, 'columns') and len(data.columns) >= 2:
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) >= 2:
                # Multiple numeric columns - scatter or correlation
                sns.scatterplot(data=data, x=numeric_cols[0], y=numeric_cols[1])
                plt.xlabel(str(numeric_cols[0]))  # MANDATORY: Always label x-axis
                plt.ylabel(str(numeric_cols[1]))  # MANDATORY: Always label y-axis
                plt.title(f"{{title}}: {{numeric_cols[0]}} vs {{numeric_cols[1]}}")
            elif len(numeric_cols) == 1:
                # Single numeric column - histogram
                data[numeric_cols[0]].hist(bins=20, alpha=0.7)
                plt.xlabel(str(numeric_cols[0]))  # MANDATORY: Always label x-axis
                plt.ylabel('Frequency')           # MANDATORY: Always label y-axis
                plt.title(f"{{title}}: Distribution of {{numeric_cols[0]}}")
            else:
                # Categorical data - bar chart
                value_counts = data.iloc[:, 0].value_counts().head(10)
                value_counts.plot(kind='bar')
                plt.xlabel(str(data.columns[0]))  # MANDATORY: Always label x-axis
                plt.ylabel('Count')               # MANDATORY: Always label y-axis
                plt.title(f"{{title}}: Count by Category")
        else:
            # Simple data - basic plot
            if hasattr(data, 'plot'):
                data.plot(kind='bar' if len(data) <= 20 else 'line')
                plt.xlabel('Index')               # MANDATORY: Always label x-axis
                plt.ylabel('Value')               # MANDATORY: Always label y-axis
            else:
                plt.bar(range(len(data)), data)
                plt.xlabel('Index')               # MANDATORY: Always label x-axis
                plt.ylabel('Value')               # MANDATORY: Always label y-axis
            plt.title(title)
        
        plt.xticks(rotation=45)
        plt.tight_layout()
        
    except Exception:
        # Fallback simple plot
        plt.plot([1, 2, 3, 4], [1, 4, 2, 3])
        plt.xlabel('X-axis')                  # MANDATORY: Always label x-axis
        plt.ylabel('Y-axis')                  # MANDATORY: Always label y-axis
        plt.title(title)
    
    # Convert to base64
    buffer = BytesIO()
    plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
    plt.close()
    
    return f"{{image_base64}}"

def safe_convert(value, target_type=float, default=0):
    '''Safely convert values to JSON-serializable types'''
    try:
        if pd.isna(value) or value is None:
            return default
        # Handle NumPy types
        if hasattr(value, 'item'):  # NumPy scalars have .item() method
            value = value.item()
        # Convert to target type
        if target_type == int:
            return int(float(value))  # Convert via float to handle strings
        elif target_type == float:
            return float(value)
        elif target_type == str:
            return str(value)
        else:
            return target_type(value)
    except (ValueError, TypeError, AttributeError):
        return default

def validate_and_clean_results(results):
    '''Ensure all results are JSON serializable'''
    cleaned = []
    for i, item in enumerate(results):
        if i == 0:  # Numeric result
            cleaned.append(safe_convert(item, int, 0))
        elif i == 1:  # String result
            cleaned.append(safe_convert(item, str, "No data"))
        elif i == 2:  # Float result
            cleaned.append(safe_convert(item, float, 0.0))
        elif i == 3:  # Base64 image
            cleaned.append(str(item) if item else "")
        else:
            cleaned.append(safe_convert(item, str, ""))
    return cleaned

def main():
    try:
        question = "{question}"
        question_lower = question.lower()
        
        # STEP 1: DETERMINE DATA SOURCE STRATEGY
        sql_keywords = ['select', 'from', 'where', 's3://', 'read_parquet', 'read_csv', 
                       'duckdb', 'install httpfs', '.parquet', 'group by', 'order by']
        web_keywords = ['wikipedia', 'scrape', 'website', 'html', 'web page', 'url']
        
        is_sql_query = any(keyword in question_lower for keyword in sql_keywords)
        is_web_scraping = any(keyword in question_lower for keyword in web_keywords) and not is_sql_query
        
        print(f"Analysis strategy: SQL={{is_sql_query}}, Web={{is_web_scraping}}")
        
        # STEP 2: DATA ACQUISITION
        if is_sql_query:
            # DuckDB approach for SQL queries and remote data
            print("Using DuckDB for data access...")
            conn = duckdb.connect()
            conn.execute("INSTALL httpfs; LOAD httpfs;")
            conn.execute("INSTALL parquet; LOAD parquet;")
            
            # Execute SQL query extracted from question
            # Extract and execute the actual SQL query from the question
            # df = conn.execute("YOUR_SQL_QUERY_HERE").fetchdf()
            
            # CRITICAL: Replace this with actual SQL execution based on question
            print("ERROR: SQL query extraction not implemented")
            raise ValueError("SQL query must be extracted from question and executed")
            
        elif is_web_scraping:
            # Web scraping approach
            print("Using web scraping...")
            headers = {{'User-Agent': 'Mozilla/5.0 (compatible; DataAnalyzer/1.0)'}}
            
            # Extract URL from question and scrape
            # CRITICAL: Extract actual URL from question and implement scraping
            print("ERROR: URL extraction and scraping not implemented")
            raise ValueError("URL must be extracted from question and scraped")
            
        else:
            # 🔒 MANDATORY WORKSPACE DATA LOADING - NO SAMPLE/FALLBACK DATA ALLOWED
            print("Loading data files from workspace - STRICT COMPLIANCE MODE")
            
            # 🚨 CRITICAL: Only load data from files present in the sandbox workspace
            # Look for data files with strict filtering - exclude system/result files
            csv_files = [f for f in Path('.').glob('*.csv') if not any(exclude in f.name.lower() for exclude in ['result', 'output', 'temp', 'log', 'error'])]
            json_files = [f for f in Path('.').glob('*.json') if not any(exclude in f.name.lower() for exclude in ['result', 'output', 'temp', 'config', 'settings', 'log', 'error'])]
            parquet_files = [f for f in Path('.').glob('*.parquet')]
            xlsx_files = [f for f in Path('.').glob('*.xlsx')]
            
            # 🎯 PRIORITY: JSON attachments are commonly named 'data.json' or similar
            all_data_files = json_files + csv_files + parquet_files + xlsx_files
            
            # Priority order for data files (JSON attachments typically use these names)
            priority_names = ['data.json', 'input.json', 'dataset.json', 'attachment.json', 'file.json', 
                            'data.csv', 'input.csv', 'dataset.csv', 'sample.csv']
            
            # Check for priority files first (likely JSON attachments)
            primary_file = None
            for priority_name in priority_names:
                if Path(priority_name).exists():
                    primary_file = Path(priority_name)
                    print(f"📋 Found priority data file: {{priority_name}}")
                    break
            
            # If no priority files, use first available data file
            if not primary_file and all_data_files:
                primary_file = all_data_files[0]
                print(f"📋 Using first available data file: {{primary_file.name}}")
            
            # 🚨 CRITICAL ERROR HANDLING: No fallback or sample data allowed
            if not primary_file:
                available_files = [f.name for f in Path('.').iterdir() if f.is_file()]
                print(f"❌ CRITICAL ERROR: No data files found in workspace")
                print(f"   Available files: {{available_files}}")
                print(f"   Expected: .csv, .json, .parquet, or .xlsx files")
                print(f"   📋 JSON attachments should be saved as data.json or similar")
                print(f"   🚨 NO SAMPLE OR FALLBACK DATA IS ALLOWED")
                raise FileNotFoundError(
                    f"No data files found in workspace. Available files: {{available_files}}. "
                    f"Please ensure data files (especially JSON attachments) are properly uploaded to the sandbox workspace. "
                    f"No sample or fallback data is permitted."
                )
            
            print(f"🔍 Loading data from workspace file: {{primary_file.name}}")
            
            # 🔒 ENHANCED DATA LOADING WITH JSON ATTACHMENT INTELLIGENCE
            try:
                if primary_file.suffix.lower() == '.json':
                    print(f"📋 Processing JSON file (likely attachment): {{primary_file.name}}")
                    try:
                        with open(primary_file, 'r', encoding='utf-8') as f:
                            json_data = json.load(f)
                        
                        print(f"✅ JSON loaded successfully. Type: {{type(json_data).__name__}}")
                        
                        # 🎯 INTELLIGENT JSON STRUCTURE HANDLING (for attachments)
                        if isinstance(json_data, list):
                            print(f"📊 JSON contains list with {{len(json_data)}} items")
                            if len(json_data) > 0 and isinstance(json_data[0], dict):
                                # List of objects - normalize to DataFrame
                                df = pd.json_normalize(json_data)
                                print(f"✅ Normalized to DataFrame: {{df.shape}}")
                            else:
                                # List of simple values
                                df = pd.DataFrame({{'value': json_data}})
                                print(f"✅ Created DataFrame from simple list")
                        elif isinstance(json_data, dict):
                            print(f"📊 JSON contains dictionary with keys: {{list(json_data.keys())}}")
                            # Check if dictionary contains lists (common in JSON attachments)
                            list_keys = [k for k, v in json_data.items() if isinstance(v, list)]
                            if list_keys:
                                print(f"📋 Found list-based keys: {{list_keys}}")
                                # Use longest list as primary data or create DataFrame from all lists
                                df = pd.DataFrame(json_data)
                                print(f"✅ Created DataFrame from dictionary with lists")
                            else:
                                # Simple key-value pairs - single row DataFrame
                                df = pd.DataFrame([json_data])
                                print(f"✅ Created single-row DataFrame from key-value pairs")
                        else:
                            # Single value - wrap in DataFrame
                            df = pd.DataFrame({{'value': [json_data]}})
                            print(f"✅ Wrapped single value in DataFrame")
                            
                    except json.JSONDecodeError as e:
                        print(f"❌ JSON parsing error for {{primary_file.name}}: {{e}}")
                        print(f"🔄 Attempting pandas read_json as fallback...")
                        try:
                            df = pd.read_json(primary_file, lines=True)  # Try JSON lines format
                            print(f"✅ Fallback successful with JSON lines format")
                        except:
                            df = pd.read_json(primary_file)  # Standard pandas JSON reader
                            print(f"✅ Fallback successful with standard JSON reader")
                    except UnicodeDecodeError as e:
                        print(f"❌ Encoding error for {{primary_file.name}}: {{e}}")
                        print(f"🔄 Trying alternative encodings...")
                        for encoding in ['latin-1', 'cp1252', 'iso-8859-1']:
                            try:
                                with open(primary_file, 'r', encoding=encoding) as f:
                                    json_data = json.load(f)
                                df = pd.json_normalize(json_data) if isinstance(json_data, list) else pd.DataFrame([json_data])
                                print(f"✅ Success with encoding: {{encoding}}")
                                break
                            except:
                                continue
                        else:
                            raise ValueError(f"Could not read JSON file {{primary_file.name}} with any encoding")
                        
                elif primary_file.suffix.lower() == '.csv':
                    print(f"📊 Processing CSV file: {{primary_file.name}}")
                    # Enhanced CSV loading with encoding detection
                    try:
                        df = pd.read_csv(primary_file, encoding='utf-8')
                        print(f"✅ CSV loaded with UTF-8 encoding")
                    except UnicodeDecodeError:
                        print(f"🔄 UTF-8 failed, trying alternative encodings...")
                        for encoding in ['latin-1', 'cp1252', 'iso-8859-1']:
                            try:
                                df = pd.read_csv(primary_file, encoding=encoding)
                                print(f"✅ CSV loaded with {{encoding}} encoding")
                                break
                            except:
                                continue
                        else:
                            raise ValueError(f"Could not read CSV file {{primary_file.name}} with any encoding")
                            
                elif primary_file.suffix.lower() == '.parquet':
                    print(f"📊 Processing Parquet file: {{primary_file.name}}")
                    df = pd.read_parquet(primary_file)
                    print(f"✅ Parquet loaded successfully")
                    
                elif primary_file.suffix.lower() in ['.xlsx', '.xls']:
                    print(f"📊 Processing Excel file: {{primary_file.name}}")
                    df = pd.read_excel(primary_file)
                    print(f"✅ Excel loaded successfully")
                    
                else:
                    print(f"⚠️  Unknown file type {{primary_file.suffix}}, attempting CSV read...")
                    df = pd.read_csv(primary_file)
                    print(f"✅ Unknown file type loaded as CSV")
                    
                # 🔍 MANDATORY DATA VALIDATION
                if df.empty:
                    raise ValueError(f"Loaded data file {{primary_file.name}} is empty. No sample or fallback data is allowed.")
                    
                print(f"✅ DATA SUCCESSFULLY LOADED FROM WORKSPACE")
                print(f"   📋 File: {{primary_file.name}}")
                print(f"   📊 Shape: {{df.shape}}")
                print(f"   📋 Columns: {{list(df.columns)}}")
                
            except Exception as e:
                print(f"❌ CRITICAL ERROR loading {{primary_file.name}}: {{str(e)}}")
                print(f"🚨 NO SAMPLE OR FALLBACK DATA IS PERMITTED")
                print(f"   The file {{primary_file.name}} exists but cannot be processed")
                print(f"   Common issues: Invalid JSON format, encoding problems, empty file")
                print(f"   For JSON attachments: Ensure valid JSON structure")
                raise ValueError(f"Failed to load data from {{primary_file.name}}: {{str(e)}}. No fallback data is allowed.")
            
            print(f"Loaded dataframe with shape: {{df.shape}}")
            print(f"Columns: {{list(df.columns)}}")
            print("First few rows:")
            print(df.head())
        
        # STEP 3: DATA EXPLORATION
        print(f"Data shape: {{df.shape}}")
        print(f"Columns: {{list(df.columns)}}")
        print("\\nFirst few rows:")
        print(df.head())
        
        # Data quality check
        missing_count = df.isnull().sum().sum()
        duplicate_count = df.duplicated().sum()
        print(f"\\nData quality: {{missing_count}} missing, {{duplicate_count}} duplicates")
        
        # STEP 4: ANALYSIS EXECUTION
        # Implement the analysis to answer each question/requirement
        print("\\nExecuting analysis...")
        
        # Extract data characteristics for analysis
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        text_cols = df.select_dtypes(include=['object']).columns
        date_cols = df.select_dtypes(include=['datetime64']).columns
        
        print(f"Numeric columns: {{list(numeric_cols)}}")
        print(f"Text columns: {{list(text_cols)}}")
        print(f"Date columns: {{list(date_cols)}}")
        
        # STEP 5: ANSWER EACH QUESTION SYSTEMATICALLY
        # Parse the question to identify specific requirements
        # THIS SECTION SHOULD BE CUSTOMIZED BASED ON THE ACTUAL QUESTION
        
        # Example analysis patterns - customize these based on your actual question:
        
        # Pattern 1: Count/How many questions
        if 'how many' in question_lower or 'count' in question_lower:
            count_result = len(df)
            print(f"Count result: {{count_result}}")
        
        # Pattern 2: Finding maximum/top/highest values  
        if 'top' in question_lower or 'highest' in question_lower or 'maximum' in question_lower:
            if len(numeric_cols) > 0:
                max_col = numeric_cols[0]  # Adjust column selection based on question
                max_value = df[max_col].max()
                max_row = df[df[max_col] == max_value].iloc[0]
                print(f"Maximum {{max_col}}: {{max_value}}")
        
        # Pattern 3: Average/mean calculations
        if 'average' in question_lower or 'mean' in question_lower:
            if len(numeric_cols) > 0:
                avg_col = numeric_cols[0]  # Adjust column selection
                avg_value = df[avg_col].mean()
                print(f"Average {{avg_col}}: {{avg_value}}")
        
        # Pattern 4: Correlation analysis
        if 'correlation' in question_lower:
            if len(numeric_cols) >= 2:
                corr_result = df[numeric_cols[0]].corr(df[numeric_cols[1]])
                print(f"Correlation: {{corr_result}}")
        
        # Pattern 5: Date-related analysis
        if 'date' in question_lower and len(date_cols) > 0:
            date_col = date_cols[0]
            earliest_date = df[date_col].min()
            latest_date = df[date_col].max()
            print(f"Date range: {{earliest_date}} to {{latest_date}}")
        
        # STEP 6: CREATE VISUALIZATIONS
        # Create appropriate visualizations based on question requirements
        print("\\nCreating visualizations...")
        
        # 🚨 CRITICAL: ALL CHARTS MUST HAVE LABELED AXES
        # MANDATORY AXIS LABELING REQUIREMENTS:
        # - Always use plt.xlabel() to label the x-axis with descriptive text
        # - Always use plt.ylabel() to label the y-axis with descriptive text
        # - Use clear, meaningful axis labels that describe the data
        # - Include units when applicable (e.g., "Sales ($)", "Temperature (°C)", "Count", "Percentage (%)")
        # - For categorical data: xlabel = category name, ylabel = "Count" or metric name
        # - For time series: xlabel = "Time" or "Date", ylabel = variable name with units
        # - For correlations: xlabel = first variable name, ylabel = second variable name
        # - Never leave axes unlabeled - even generic labels like "X-axis", "Y-axis" are better than none
        # 
        # CHART-SPECIFIC LABELING EXAMPLES:
        # - Bar charts: plt.xlabel("Region"), plt.ylabel("Sales ($)")
        # - Line charts: plt.xlabel("Date"), plt.ylabel("Revenue ($)")
        # - Histograms: plt.xlabel("Value Range"), plt.ylabel("Frequency")
        # - Scatter plots: plt.xlabel("Variable 1"), plt.ylabel("Variable 2")
        
        # Create base visualization - customize based on specific requirements
        plt.figure(figsize=(10, 6))
        
        # Example visualization patterns:
        if len(numeric_cols) >= 2:
            # Scatter plot for numeric relationships
            plt.scatter(df[numeric_cols[0]], df[numeric_cols[1]], alpha=0.6)
            plt.xlabel(numeric_cols[0])  # MANDATORY: Always label x-axis
            plt.ylabel(numeric_cols[1])  # MANDATORY: Always label y-axis
            plt.title(f"{{numeric_cols[0]}} vs {{numeric_cols[1]}}")
        elif len(numeric_cols) == 1:
            # Histogram for single numeric column
            df[numeric_cols[0]].hist(bins=20, alpha=0.7, color='blue')
            plt.xlabel(numeric_cols[0])  # MANDATORY: Always label x-axis
            plt.ylabel('Frequency')     # MANDATORY: Always label y-axis
            plt.title(f"Distribution of {{numeric_cols[0]}}")
        else:
            # Bar chart for categorical data
            if len(text_cols) > 0:
                value_counts = df[text_cols[0]].value_counts().head(10)
                value_counts.plot(kind='bar', color='green')
                plt.xlabel(text_cols[0])  # MANDATORY: Always label x-axis
                plt.ylabel('Count')       # MANDATORY: Always label y-axis
                plt.title(f"Count by {{text_cols[0]}}")
            else:
                # Error: Cannot create meaningful visualization without data
                plt.text(0.5, 0.5, 'No suitable data for visualization', 
                        horizontalalignment='center', verticalalignment='center', 
                        transform=plt.gca().transAxes, fontsize=12)
                plt.xlabel('X-axis')      # MANDATORY: Even for error cases
                plt.ylabel('Y-axis')      # MANDATORY: Even for error cases
                plt.title("No Data Available")
        
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # Convert to base64
        buffer = BytesIO()
        plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        plt.close()
        
        # Create the base64 image string (no prefix)
        image_data_uri = f"{{image_base64}}"
        
        # STEP 7: COMPILE RESULTS IN REQUIRED FORMAT
        # 🔒 CRITICAL SECTION - MUST FOLLOW EXACT OUTPUT STRUCTURE FROM QUESTION
        
        print("\\nCompiling results in STRICT compliance with question format...")
        
        # ============================================================================
        # ⚠️  REPLACE THIS SECTION WITH QUESTION-SPECIFIC IMPLEMENTATION
        # ⚠️  FOLLOW THE OUTPUT FORMAT INSTRUCTIONS EXACTLY
        # ⚠️  DO NOT DEVIATE FROM THE SPECIFIED STRUCTURE
        # ============================================================================
        
        # Determine the exact output format required by the question
        question_lower = question.lower()
        
        # FORMAT DETECTION AND COMPLIANCE
        if "json object with keys" in question_lower or "return a json object" in question_lower:
            # 🔒 JSON OBJECT FORMAT - Extract exact keys from question and implement
            print("🔍 Detected JSON OBJECT format requirement")
            print("📋 Must implement exact key structure from question using real data")
            
            # 🚨 CRITICAL: This section must be customized based on the specific question
            # The LLM must extract the required keys from the question and implement each one
            # using actual data from the workspace - NO SAMPLE DATA OR PLACEHOLDER VALUES
            
            result = {{}}
            
            # 🔍 EXTRACT REQUIRED KEYS FROM QUESTION AND IMPLEMENT EACH ONE:
            # Parse the question to find "Return a JSON object with keys:" section
            # For each key specified (e.g., "- `edge_count`: number"), implement calculation
            # Use only real data from the loaded dataframe/workspace files
            
            # IMPLEMENTATION PATTERN (customize for actual question):
            # result["key_name"] = safe_convert(your_calculation_from_real_data, target_type, default_value)
            
            print("🚨 CRITICAL: The LLM must implement the specific keys required by this question")
            print("   Each key must be calculated from real data in the workspace")
            print("   NO sample, placeholder, or fallback data is permitted")
            print(f"   📋 Question requiring implementation: {{question}}")
            
            # This will be replaced by LLM with actual implementation
            raise ValueError(
                f"JSON object format detected. The LLM must implement the specific keys "
                f"required by the question using real data from workspace files. "
                f"Question: '{{question}}'. No sample data allowed."
            )
            
        elif "json array" in question_lower and ("string" in question_lower or "strings" in question_lower):
            # 🔒 JSON ARRAY OF STRINGS FORMAT
            print("🔍 Detected JSON ARRAY OF STRINGS format requirement")
            
            # Count numbered questions to determine exact array size
            import re
            numbered_questions = re.findall(r'^\\s*\\d+\\.\\s', question, re.MULTILINE)
            expected_elements = len(numbered_questions)
            print(f"📋 Must return exactly {{expected_elements}} string elements")
            
            # 🚨 CRITICAL: Each array element must answer the corresponding numbered question
            # using real data from the workspace - NO SAMPLE OR PLACEHOLDER ANSWERS
            
            results = []
            
            # 🔍 IMPLEMENT ANSWERS FOR EACH NUMBERED QUESTION:
            # Parse each numbered question (1., 2., 3., etc.) from the question text
            # Calculate the answer using real data from the loaded dataframe/workspace files
            # Convert all answers to strings as required by the format
            
            print("🚨 CRITICAL: The LLM must implement answers for each numbered question")
            print("   Each answer must be calculated from real data in the workspace")
            print("   All answers must be converted to strings")
            print("   NO sample, placeholder, or fallback answers are permitted")
            print(f"   📋 Expected {{expected_elements}} answers for questions 1-{{expected_elements}}")
            print(f"   🔍 Question text: {{question}}")
            
            # This will be replaced by LLM with actual implementation for each question
            for i in range(expected_elements):
                if i == expected_elements - 1 and "visualization" in question_lower:
                    # Last element might be a visualization if mentioned in question
                    results.append("")  # Will be replaced with actual image
                else:
                    # Each specific answer must be implemented by the LLM
                    raise ValueError(
                        f"Array element {{i+1}} not implemented. The LLM must calculate "
                        f"the answer to question {{i+1}} using real data from workspace files. "
                        f"No sample or placeholder answers allowed."
                    )
            
            result = results
            
        else:
            # � ERROR: Invalid format fallback not allowed
            print("❌ CRITICAL ERROR: Unable to determine required output format from question")
            print("   The question must specify either:")
            print("   1. 'JSON object with keys:' followed by key specifications")  
            print("   2. 'JSON array of strings' with numbered questions")
            print("   No fallback or sample data is allowed")
            
            raise ValueError(
                f"Unable to determine required output format from question: '{question}'. "
                f"The question must clearly specify either 'JSON object with keys:' followed by key list, "
                f"or 'JSON array of strings' with numbered questions. "
                f"No default formats or sample data are allowed."
            )
        
        # ============================================================================
        # 🔒 MANDATORY VALIDATION - VERIFY STRUCTURE COMPLIANCE
        # ============================================================================
        
        print(f"\\n🔍 VALIDATION: Final result structure compliance check")
        print(f"   Type: {{type(result).__name__}}")
        
        if isinstance(result, dict):
            print(f"   JSON Object with {{len(result)}} keys: {{list(result.keys())}}")
            # Validate object structure
            if not result:
                print("   ❌ ERROR: Empty object")
                result = {{"error": "No data processed"}}
        elif isinstance(result, list):
            print(f"   JSON Array with {{len(result)}} elements")
            # Validate array structure  
            if not result:
                print("   ❌ ERROR: Empty array")
                result = ["No data"]
            # Ensure string elements for array format
            if "json array" in question_lower and "string" in question_lower:
                result = [str(item) for item in result]
                print(f"   ✅ Converted all elements to strings")
        else:
            print(f"   ❌ ERROR: Invalid type {{type(result).__name__}}")
            result = ["Error: Invalid result type"]
        
        print(f"\\n✅ Final result ready for JSON serialization")
        
        # STEP 8: SAVE RESULTS WITH ROBUST JSON ENCODING
        with open('result.json', 'w') as f:
            json.dump(result, f, cls=RobustJSONEncoder, ensure_ascii=False, indent=2)
        
        print("Analysis completed successfully!")
        print(f"Results saved to result.json")
        
    except Exception as e:
        print(f"Error: {{e}}")
        import traceback
        traceback.print_exc()
        
        # 🚨 CRITICAL ERROR OCCURRED - NO SAMPLE DATA FALLBACK
        print(f"❌ CRITICAL ERROR in analysis: {{str(e)}}")
        print(f"🔍 Question was: {{question}}")
        print("📋 NO SAMPLE OR FALLBACK DATA IS PERMITTED")
        print("   All results must be calculated from real data in workspace")
        import traceback
        traceback.print_exc()
        
        # Error handling - must still respect output format detected from question
        question_lower = question.lower()
        if "json object with keys" in question_lower:
            # Return error object with detected keys (no sample data)
            error_result = {{"error": f"Analysis failed: {{str(e)}}", "data_source": "error_no_data"}}
        elif "json array" in question_lower and "string" in question_lower:
            # Return error array with proper length (no sample data)
            import re
            numbered_questions = re.findall(r'^\\s*\\d+\\.\\s', question, re.MULTILINE)
            expected_elements = len(numbered_questions) if numbered_questions else 1
            error_result = [f"Error: {{str(e)}}" for _ in range(expected_elements)]
        else:
            # Unknown format - minimal error response
            error_result = [f"Error: {{str(e)}}"]
        
        # Save error result (no fallback data)
        with open('result.json', 'w') as f:
            json.dump(error_result, f, cls=RobustJSONEncoder, ensure_ascii=False, indent=2)
            
        print(f"💾 Error result saved - no sample data used")

if __name__ == "__main__":
    main()
```

**COMPREHENSIVE CHART CREATION REQUIREMENTS:**

🎯 **MANDATORY AXIS LABELING FOR ALL CHART TYPES:**
- **Bar Charts:** plt.xlabel("Category Name"), plt.ylabel("Value/Count/Metric ($)")
- **Line Charts:** plt.xlabel("Time/Date"), plt.ylabel("Variable Name (units)")  
- **Scatter Plots:** plt.xlabel("X Variable Name"), plt.ylabel("Y Variable Name")
- **Histograms:** plt.xlabel("Variable Name"), plt.ylabel("Frequency/Count")
- **Box Plots:** plt.xlabel("Category"), plt.ylabel("Variable Name (units)")
- **Heatmaps:** Include axis labels even if using seaborn (plt.xlabel/ylabel still required)

📊 **CHART TYPE REQUIREMENTS:**
- Always choose appropriate chart types based on data and question
- Use color specifications when requested (e.g., "blue bars", "red line")
- Include proper titles that describe what the chart shows
- Apply plt.tight_layout() to prevent label cutoff
- Use appropriate figure size: plt.figure(figsize=(10, 6)) or larger for complex charts

⚠️ **CRITICAL CHART VALIDATION:**
- NEVER create charts without axis labels
- ALWAYS test that plt.xlabel() and plt.ylabel() are called
- Include units in parentheses when applicable: "Sales ($)", "Temperature (°C)"
- For time series, format dates properly and rotate labels if needed
- Use descriptive titles that explain the visualization

**SPECIFIC GUIDANCE BY ANALYSIS TYPE:**

{get_analysis_type_guidance(analysis_type)}

**CRITICAL REMINDERS:**
1. ALWAYS use RobustJSONEncoder when saving JSON
2. ALWAYS use safe_convert() for numeric conversions  
3. ALWAYS use validate_and_clean_results() before saving
4. NEVER save raw pandas/numpy objects directly to JSON
5. Handle NaN, inf, and None values explicitly

Generate the Python code now:
"""

    # Enhance the prompt with context-specific examples and guidance
    enhanced_prompt = enhance_prompt_with_context(prompt, question, analysis_type)
    
    return enhanced_prompt

def get_analysis_type_guidance(analysis_type: str) -> str:
    """Get specific guidance based on analysis type."""
    guidance = {
        "statistical": """
**STATISTICAL ANALYSIS FOCUS:**
- Calculate comprehensive descriptive statistics (mean, median, std, quartiles, skewness, kurtosis)
- Perform correlation analysis for numeric variables (use df.corr())
- Create distribution plots (histograms, box plots, violin plots)
- Apply statistical tests when appropriate (t-tests, chi-square, normality tests)
- Check for outliers using IQR method or z-scores
- Generate summary statistics tables and interpretation
- **JSON Safety:** Use safe_convert() for all statistical calculations to handle NaN/inf values

**MANDATORY CHART REQUIREMENTS:**
- ALWAYS use plt.xlabel() with descriptive labels (e.g., "Variable Name", "Sales Amount ($)")
- ALWAYS use plt.ylabel() with descriptive labels (e.g., "Frequency", "Count", "Probability")
- Include units in axis labels when applicable (e.g., "Temperature (°C)", "Revenue ($)", "Time (hours)")
- For histograms: xlabel = variable name, ylabel = "Frequency" or "Count"
- For scatter plots: xlabel = first variable, ylabel = second variable
- For box plots: xlabel = category, ylabel = variable name with units

**Key outputs:** Count of variables, strongest correlation coefficient, primary distribution characteristic
        """,
        
        "network": """
**NETWORK ANALYSIS FOCUS:**
- Use networkx library for graph construction and analysis
- Calculate key network metrics: degree centrality, betweenness centrality, clustering coefficient
- Identify communities using community detection algorithms
- Create network visualizations with node sizing based on importance
- Analyze network properties: diameter, density, average path length
- Find influential nodes and network components
- **JSON Safety:** Convert all networkx metrics using safe_convert() as they often return numpy types

**MANDATORY CHART REQUIREMENTS:**
- For network graphs: xlabel = "X Position", ylabel = "Y Position" (or geographic coordinates if applicable)
- For degree distribution plots: xlabel = "Degree", ylabel = "Number of Nodes"
- For centrality plots: xlabel = "Node Index" or "Node Name", ylabel = "Centrality Score"
- For community plots: xlabel = "Community", ylabel = "Number of Nodes"
- Always include clear axis labels even for network visualizations

**Key outputs:** Number of nodes/edges, most central node, average clustering coefficient
        """,
        
        "timeseries": """
**TIME SERIES ANALYSIS FOCUS:**
- Parse datetime columns using pd.to_datetime() with appropriate format
- Set datetime as index for time series operations
- Calculate trends using rolling averages and linear regression
- Identify seasonality patterns using decomposition
- Create time series plots with trend lines and seasonal components
- Calculate year-over-year or period-over-period growth rates
- Handle missing time periods and irregular intervals
- **JSON Safety:** Convert datetime objects to ISO format strings, handle pandas Timestamp objects

**MANDATORY CHART REQUIREMENTS:**
- For time series plots: xlabel = "Date" or "Time Period", ylabel = variable name with units
- For trend analysis: xlabel = "Time", ylabel = "Trend Value" or specific metric
- For seasonal decomposition: xlabel = "Time Period", ylabel = component name (e.g., "Seasonal Component")
- For autocorrelation plots: xlabel = "Lag", ylabel = "Autocorrelation"
- Always format time axis labels for readability (e.g., plt.xticks(rotation=45))

**Key outputs:** Total time periods, strongest trend direction, peak/trough values
        """,
        
        "ml": """
**MACHINE LEARNING ANALYSIS FOCUS:**
- Perform comprehensive data preprocessing (scaling, encoding, feature selection)
- Apply appropriate ML techniques: clustering (KMeans), classification, regression
- Create feature importance plots and correlation matrices
- Calculate and report performance metrics (accuracy, precision, recall, R²)
- Use cross-validation for robust model evaluation
- Visualize decision boundaries or cluster separations
- Handle categorical variables with proper encoding
- **JSON Safety:** All sklearn metrics return numpy types - always use safe_convert()

**MANDATORY CHART REQUIREMENTS:**
- For feature importance: xlabel = "Features", ylabel = "Importance Score"
- For clustering plots: xlabel = "First Principal Component" or feature name, ylabel = "Second Principal Component" or feature name
- For performance metrics: xlabel = "Model", ylabel = "Score" or specific metric name
- For correlation matrices: xlabel = "Features", ylabel = "Features" (with heatmap)
- For learning curves: xlabel = "Training Set Size" or "Epochs", ylabel = "Accuracy" or "Loss"

**Key outputs:** Number of features, best model performance score, most important feature
        """,
        
        "database": """
**DATABASE ANALYSIS FOCUS:**
- Use DuckDB for SQL query execution on local and remote data sources
- Install and configure necessary extensions (httpfs for S3/HTTP, parquet for .parquet files)
- Execute complex SQL queries with proper JOIN, GROUP BY, and aggregation functions
- Handle large datasets efficiently using DuckDB's columnar processing
- Convert query results to pandas DataFrames for further analysis
- Implement proper error handling for database connections and queries
- Use DuckDB-specific functions and avoid SQLite/MySQL syntax differences
- **JSON Safety:** DuckDB results may contain numpy dtypes - always use RobustJSONEncoder

**Key outputs:** Query result count, aggregated metric, database connection status
        """,
        
        "json": """
**JSON DATA ANALYSIS FOCUS:**
- **Intelligent Structure Detection:**
  - Automatically identify JSON structure (array of objects, nested dictionaries, key-value pairs)
  - Use pd.json_normalize() for complex nested structures
  - Handle both single-level and multi-level JSON hierarchies
  
- **Robust Data Loading:**
  - Try multiple encoding formats (utf-8, latin-1, cp1252) for file reading
  - Implement graceful fallbacks if pandas.read_json() fails
  - Handle edge cases like empty files, malformed JSON, or mixed data types
  
- **Smart Data Transformation:**
  - Flatten nested objects when beneficial for analysis
  - Preserve important hierarchical relationships
  - Convert date strings to datetime objects automatically
  - Handle missing values in JSON arrays appropriately
  
- **JSON-Optimized Analysis:**
  - Leverage schema flexibility to extract meaningful patterns
  - Analyze both structured fields and text content within JSON
  - Create visualizations that highlight JSON data characteristics
  - Identify data quality issues specific to JSON format
  
- **Best Practices:**
  - Always validate JSON structure before processing
  - Use vectorized operations for large JSON datasets
  - Maintain data integrity during normalization processes
  - Provide clear feedback about data structure discovered
  
- **CRITICAL JSON Safety for Output:**
  - JSON input files may create nested numpy/pandas objects
  - Always use RobustJSONEncoder for final result serialization
  - Handle deeply nested structures that may contain non-serializable types
  - Test JSON output serialization before saving

**Key outputs:** Records count, primary data insight, calculated JSON-specific metric
        """,
        
        "general": """
**GENERAL DATA ANALYSIS FOCUS:**
- **Data Source Strategy:**
  - **SQL/DuckDB:** Use for database queries, S3/HTTP URLs, .parquet/.csv remote files
  - **Web Scraping:** Only for explicit HTML content extraction (Wikipedia, web tables)
  - **Local Files:** Default approach for uploaded CSV/JSON/Parquet files
  
- **JSON Data Handling Excellence:**
  - **Structure Recognition:** Automatically detect list vs dictionary vs nested JSON structures
  - **Smart Normalization:** Use pd.json_normalize() for nested objects, handle arrays intelligently
  - **Encoding Resilience:** Handle UTF-8, latin-1, and other character encodings gracefully
  - **Format Flexibility:** Process various JSON formats (arrays of objects, key-value pairs, nested structures)
  - **Error Recovery:** Implement fallback strategies for malformed JSON data
  
- **Analysis Pipeline:**
  1. Data discovery and quality assessment with JSON-specific considerations
  2. Exploratory data analysis with appropriate visualizations for JSON-derived data
  3. Pattern identification across potentially nested/complex data structures
  4. Meaningful summary and interpretation of hierarchical information
  
- **Robust Implementation:**
  - Handle missing data with appropriate strategies (imputation, removal)
  - Validate data types and perform necessary conversions for JSON fields
  - Create informative visualizations that match the data characteristics
  - Flatten nested structures when necessary for analysis
  - Provide clear, actionable insights based on the analysis
  - Ensure output format compliance (4-element JSON array)
  
- **MANDATORY JSON Serialization Safety:**
  - ALWAYS use RobustJSONEncoder for ALL JSON operations
  - ALWAYS use safe_convert() for numeric values from pandas/numpy
  - ALWAYS use validate_and_clean_results() before saving results
  - Handle edge cases: NaN, infinity, numpy.int64, pandas.Int64, etc.
  - Test JSON serialization in development to catch type errors early

**Key outputs:** Dataset size, primary insight, calculated metric
        """
    }
    
    return guidance.get(analysis_type, guidance["general"])

def extract_code_from_response(response_text: str) -> Optional[str]:
    """
    Extract Python code from the LLM response.
    
    Args:
        response_text: Raw response from the LLM
        
    Returns:
        Extracted Python code or None if not found
    """
    # Look for code blocks
    import re
    
    # Try to find Python code blocks
    python_pattern = r'```python\s*(.*?)\s*```'
    matches = re.findall(python_pattern, response_text, re.DOTALL)
    
    if matches:
        return matches[0].strip()
    
    # Try generic code blocks
    code_pattern = r'```\s*(.*?)\s*```'
    matches = re.findall(code_pattern, response_text, re.DOTALL)
    
    if matches:
        # Check if it looks like Python code
        code = matches[0].strip()
        if any(keyword in code for keyword in ['import ', 'def ', 'if __name__']):
            return code
    
    # If no code blocks found, try to extract code-like content
    lines = response_text.split('\n')
    code_lines = []
    in_code = False
    
    for line in lines:
        if any(keyword in line for keyword in ['import ', 'def ', 'try:', 'if ', 'for ', 'while ']):
            in_code = True
        
        if in_code:
            code_lines.append(line)
    
    if code_lines:
        return '\n'.join(code_lines)
    
    return None

def get_duckdb_examples_for_question(question: str) -> str:
    """
    Generate relevant DuckDB examples based on question content.
    
    Args:
        question: User's question text
        
    Returns:
        String with relevant DuckDB code examples
    """
    question_lower = question.lower()
    examples = []
    
    if 's3://' in question_lower:
        examples.append("""
# S3 data access example:
conn.execute("INSTALL httpfs; LOAD httpfs;")
df = conn.execute("SELECT * FROM read_parquet('s3://bucket/path/file.parquet?s3_region=us-east-1')").fetchdf()
""")
    
    if 'https://' in question_lower and ('.csv' in question_lower or '.parquet' in question_lower):
        examples.append("""
# HTTPS data access example:
conn.execute("INSTALL httpfs; LOAD httpfs;")
df = conn.execute("SELECT * FROM read_csv('https://example.com/data.csv')").fetchdf()
""")
    
    if 'date' in question_lower or 'time' in question_lower:
        examples.append("""
# Date handling examples:
# Use DATE_DIFF for date differences:
SELECT DATE_DIFF('day', start_date, end_date) as days_between FROM table
# Cast strings to dates:
SELECT column::DATE as date_col FROM table WHERE date_col > '2023-01-01'::DATE
""")
    
    if 'count' in question_lower or 'group' in question_lower:
        examples.append("""
# Aggregation examples:
SELECT category, COUNT(*) as count, AVG(value) as avg_value 
FROM table GROUP BY category ORDER BY count DESC
""")
    
    return '\n'.join(examples) if examples else ""

def validate_output_format_requirements(question: str, analysis_type: str) -> str:
    """
    Generate specific output format guidance based on question and analysis type.
    
    Args:
        question: User's question text
        analysis_type: Type of analysis being performed
        
    Returns:
        Specific guidance for output format
    """
    question_lower = question.lower()
    
    numeric_guidance = "Element 1 (numeric): "
    string_guidance = "Element 2 (string): "
    float_guidance = "Element 3 (float): "
    
    # Customize based on question type
    if 'count' in question_lower or 'how many' in question_lower:
        numeric_guidance += "Total count (integer) - use safe_convert(value, int, 0)"
        string_guidance += "Name of what was counted - use safe_convert(value, str, 'No data')"
        float_guidance += "Count as decimal - use safe_convert(value, float, 0.0)"
    elif 'top' in question_lower or 'best' in question_lower or 'maximum' in question_lower:
        numeric_guidance += "Rank or position (1, 2, 3...) - use safe_convert(value, int, 0)"
        string_guidance += "Name/identifier of top item - use safe_convert(value, str, 'Unknown')"
        float_guidance += "Value/score of top item - use safe_convert(value, float, 0.0)"
    elif 'average' in question_lower or 'mean' in question_lower:
        numeric_guidance += "Sample size or data points - use safe_convert(len(df), int, 0)"
        string_guidance += "Variable name being averaged - use safe_convert(col_name, str, 'Unknown')"
        float_guidance += "Calculated average value - use safe_convert(avg_result, float, 0.0)"
    elif 'correlation' in question_lower or 'relationship' in question_lower:
        numeric_guidance += "Number of variables analyzed - use safe_convert(num_vars, int, 0)"
        string_guidance += "Names of correlated variables - use safe_convert(f'{var1} vs {var2}', str, 'Unknown')"
        float_guidance += "Correlation coefficient - use safe_convert(corr_value, float, 0.0)"
    else:
        # General guidance
        numeric_guidance += "Primary count/ID/rank - use safe_convert(value, int, 0)"
        string_guidance += "Main categorical result/name - use safe_convert(value, str, 'No data')"
        float_guidance += "Key calculated metric - use safe_convert(value, float, 0.0)"
    
    return f"""
**CRITICAL JSON SERIALIZATION OUTPUT FORMAT:**
{numeric_guidance}
{string_guidance}
{float_guidance}
Element 4 (image): Base64 PNG string (no prefix) - ensure it's a string

**EXAMPLES WITH JSON SAFETY:**
- Count: [safe_convert(247, int, 0), safe_convert("products", str, "No data"), safe_convert(247.0, float, 0.0), image_str]
- Top item: [safe_convert(1, int, 0), safe_convert("iPhone 14", str, "Unknown"), safe_convert(999.99, float, 0.0), image_str]
- Average: [safe_convert(len(df), int, 0), safe_convert("price", str, "Unknown"), safe_convert(avg_val, float, 0.0), image_str]
- Correlation: [safe_convert(2, int, 0), safe_convert("price vs sales", str, "Unknown"), safe_convert(corr_val, float, 0.0), image_str]

**MANDATORY STEPS BEFORE JSON.DUMP():**
1. Calculate raw results
2. Apply safe_convert() to each element
3. Use validate_and_clean_results() on the array
4. Save with RobustJSONEncoder
"""

def enhance_prompt_with_context(base_prompt: str, question: str, analysis_type: str) -> str:
    """
    Enhance the base prompt with context-specific examples and strict format compliance.
    
    Args:
        base_prompt: Base prompt template
        question: User's question
        analysis_type: Type of analysis
        
    Returns:
        Enhanced prompt with specific examples and format enforcement
    """
    # Extract output format for compliance checking
    output_structure = extract_output_structure_from_question(question)
    
    # Add DuckDB examples if relevant
    duckdb_examples = get_duckdb_examples_for_question(question)
    if duckdb_examples:
        duckdb_section = f"\n**RELEVANT DUCKDB EXAMPLES:**{duckdb_examples}\n"
        base_prompt = base_prompt.replace("**8. Error Handling", duckdb_section + "**8. Error Handling")
    
    # Add specific output format guidance
    output_guidance = validate_output_format_requirements(question, analysis_type)
    base_prompt = base_prompt.replace("**STRUCTURED CODE TEMPLATE:**", 
                                      output_guidance + "\n**STRUCTURED CODE TEMPLATE:**")
    
    # Add format-specific validation code to the template
    format_validation = generate_format_validation_code(output_structure, question)
    base_prompt = base_prompt.replace(
        "with open('result.json', 'w') as f:",
        format_validation + "\n        with open('result.json', 'w') as f:"
    )
    
    # Add question-specific hints in the code template
    question_hint = f"# 🔒 QUESTION FORMAT COMPLIANCE: {question}\n        # 🚨 CRITICAL: Detect and follow EXACT output structure from question\n        # 🎯 Focus on: "
    
    if 'count' in question.lower():
        question_hint += "counting and aggregation with format compliance"
    elif 'top' in question.lower() or 'best' in question.lower():
        question_hint += "finding maximum/top values with structure validation"
    elif 'correlation' in question.lower():
        question_hint += "calculating relationships with output format adherence"
    elif 'trend' in question.lower() or 'change' in question.lower():
        question_hint += "analyzing trends with format specification compliance"
    else:
        question_hint += "comprehensive data analysis with strict format compliance"
    
    base_prompt = base_prompt.replace("def main():", f"def main():\n        {question_hint}")
    
    return base_prompt

def generate_format_validation_code(output_structure: Dict[str, Any], question: str) -> str:
    """
    Generate validation code to ensure output format compliance.
    
    Args:
        output_structure: Output structure information
        question: Original question
        
    Returns:
        Python code for format validation
    """
    if output_structure["format_type"] == "object":
        required_keys = [k["name"] for k in output_structure["required_keys"]]
        return f"""
        # 🔒 MANDATORY OBJECT FORMAT VALIDATION
        print("\\n🔍 Validating JSON object format compliance...")
        if not isinstance(result, dict):
            print("❌ CRITICAL ERROR: Result must be a dictionary (JSON object)")
            result = {{"error": "Format violation - expected JSON object"}}
        
        required_keys = {required_keys}
        missing_keys = [k for k in required_keys if k not in result]
        extra_keys = [k for k in result.keys() if k not in required_keys]
        
        if missing_keys:
            print(f"❌ MISSING REQUIRED KEYS: {{missing_keys}}")
            for key in missing_keys:
                result[key] = "Missing data"
                
        if extra_keys:
            print(f"⚠️  EXTRA KEYS DETECTED: {{extra_keys}} - removing...")
            for key in extra_keys:
                del result[key]
        
        print(f"✅ JSON object validation passed: {{len(result)}} keys")
        """
    
    elif output_structure["format_type"] == "array":
        expected_length = output_structure.get("array_elements", 4)
        return f"""
        # 🔒 MANDATORY ARRAY FORMAT VALIDATION
        print("\\n🔍 Validating JSON array format compliance...")
        if not isinstance(result, list):
            print("❌ CRITICAL ERROR: Result must be a list (JSON array)")
            result = ["Format violation - expected JSON array"]
        
        expected_length = {expected_length}
        if len(result) != expected_length:
            print(f"❌ WRONG ARRAY LENGTH: {{len(result)}} instead of {{expected_length}}")
            # Fix length
            while len(result) < expected_length:
                result.append("No data")
            result = result[:expected_length]
        
        # Ensure all elements are strings for string array format
        if "string" in question.lower():
            result = [str(item) for item in result]
            print(f"✅ Converted all {{len(result)}} elements to strings")
        
        print(f"✅ JSON array validation passed: {{len(result)}} elements")
        """
    
    else:
        return """
        # 🔒 BASIC FORMAT VALIDATION
        print("\\n🔍 Validating basic format compliance...")
        if not isinstance(result, (list, dict)):
            print("❌ ERROR: Result must be list or dict")
            result = ["Format error"]
        print(f"✅ Basic validation passed: {{type(result).__name__}}")
        """