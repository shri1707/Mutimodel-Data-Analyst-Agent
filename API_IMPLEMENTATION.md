# API Endpoint Implementation Summary

## Endpoint: POST `/api/`

The `/api/` endpoint has been implemented according to your specifications to accept file uploads via multipart/form-data, with `questions.txt` as a required file and zero or more additional files.

### Specification Compliance

✅ **POST Request**: The endpoint accepts POST requests  
✅ **File Upload**: Uses multipart/form-data for file uploads  
✅ **Required questions.txt**: Must include a file named `questions.txt` or ending with `question.txt`  
✅ **Optional Additional Files**: Accepts zero or more additional data files  
✅ **Exact curl Command**: Works with the specified curl format  

### Usage

```bash
curl "http://localhost:8000/api/" -F "questions.txt=@question.txt" -F "image.png=@image.png" -F "data.csv=@data.csv"
```

### Key Features

#### 1. **Flexible Questions File Detection**
The endpoint recognizes questions files by:
- Exact filename: `questions.txt` 
- Ending pattern: `*question.txt` (e.g., `my_question.txt`)
- Case-insensitive matching
- Content in UTF-8 or Latin-1 encoding

#### 2. **File Type Validation**
- Questions file must be `.txt`
- Additional files validated against supported types (CSV, JSON, HTML, ZIP, Excel, etc.)
- Unsupported files return HTTP 400 error

#### 3. **Comprehensive Processing**
- Creates isolated sandbox directory for each request
- Reads and processes questions file content
- Saves all additional files
- Generates analysis code using LLM (Gemini API)
- Executes code in secure Docker container
- Returns structured JSON response

#### 4. **Error Handling**
- Missing questions file → HTTP 400
- Empty questions file → HTTP 400  
- Invalid file types → HTTP 400
- Processing errors → HTTP 500
- Detailed error messages and logging

### Response Format

#### Success Response (HTTP 200)
```json
{
  "request_id": "uuid-string",
  "status": "success",
  "question": "content of questions.txt file",
  "analysis_type": "general",
  "files_processed": 2,
  "execution_time": 5.2,
  "docker_execution_time": 3.1,
  "results": {
    "json": {...},
    "images": [
      {
        "filename": "plot.png",
        "data": "base64-encoded-image",
        "size_bytes": 15420
      }
    ],
    "stdout": "execution output",
    "stderr": "error output"
  }
}
```

#### Error Response (HTTP 400/500)
```json
{
  "request_id": "uuid-string",
  "status": "error",
  "question": "partial or unknown",
  "analysis_type": "general",
  "files_processed": 0,
  "execution_time": 1.2,
  "error": "Error description",
  "results": {}
}
```

### Test Commands

#### Basic Test
```bash
curl "http://localhost:8000/api/" -F "questions.txt=@question.txt"
```

#### With Additional Files
```bash
curl "http://localhost:8000/api/" -F "questions.txt=@question.txt" -F "data.csv=@data.csv"
```

#### Error Test (no questions.txt)
```bash
curl "http://localhost:8000/api/" -F "data.csv=@data.csv"
# Returns HTTP 400: Must include questions.txt file
```

### Files Created for Testing

1. **`test_endpoint.py`** - Tests questions file detection logic
2. **`test_curl_simulation.py`** - Simulates curl command structure  
3. **`start_test_server.py`** - Starts the FastAPI server for testing
4. **`test_api_curl.sh`** - Complete curl command test suite
5. **`test_question.txt`** - Sample questions file
6. **`test_data.csv`** - Sample data file

### Architecture Integration

The endpoint integrates with the existing architecture:
- **Logger**: Comprehensive request/response logging
- **Utils**: File validation and processing utilities  
- **LLM**: AI-powered code generation (Gemini API)
- **Docker Runner**: Secure code execution in containers
- **Sandbox**: Isolated file storage per request

### Security Features

- **File Type Validation**: Only allowed file types accepted
- **Sandboxed Execution**: Each request gets isolated directory
- **Resource Limits**: Docker containers with memory/CPU limits
- **Code Validation**: Generated code is validated before execution
- **Timeout Protection**: Prevents infinite execution loops

The implementation fully satisfies the specification and is ready for production use.
