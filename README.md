# Da## Features

- 🚀 **FastAPI Backend**: High-performance async API with automatic OpenAPI documentation
- 🤖 **AI Code Generation**: Uses Google's Gemini API to generate analysis code from natural language
- 🐳 **Secure Execution**: Runs generated code in isolated Docker containers
- 📊 **Multiple Analysis Types**: Statistical, network analysis, time series, machine learning
- 📁 **File Support**: CSV, JSON, HTML, ZIP, Excel, TXT, and more
- 📈 **Chart Standards**: All generated charts include properly labeled axes with descriptive text and units
- 🖼️ **Image Output**: API returns clean base64-encoded images for easy integration
- 📝 **Comprehensive Logging**: Detailed request and execution logging
- 🔒 **Security**: Sandboxed execution with resource limits and code validation API

A FastAPI-based backend system for file-driven data analysis with AI-powered code generation. Users can upload data files (CSV, JSON, HTML, ZIP, TXT, etc.) along with natural language questions, and the system generates and executes Python analysis code in secure Docker containers.

## Features

- 🚀 **FastAPI Backend**: High-performance async API with automatic OpenAPI documentation
- 🤖 **AI Code Generation**: Uses Google's Gemini API to generate analysis code from natural language
- 🐳 **Secure Execution**: Runs generated code in isolated Docker containers
- 📊 **Multiple Analysis Types**: Statistical, network analysis, time series, machine learning
- 📁 **File Support**: CSV, JSON, HTML, ZIP, Excel, TXT, and more
- � **Chart Standards**: All generated charts include properly labeled axes with descriptive text and units
- 🖼️ **Image Output**: API returns clean base64-encoded images for easy integration
- �📝 **Comprehensive Logging**: Detailed request and execution logging
- 🔒 **Security**: Sandboxed execution with resource limits and code validation

## Architecture

```
┌─────────────────┐    ┌──────────────┐    ┌─────────────────┐
│   File Upload   │───▶│  FastAPI     │───▶│   Gemini API    │
│   + Question    │    │  Backend     │    │  (Code Gen)     │
└─────────────────┘    └──────────────┘    └─────────────────┘
                              │                      │
                              ▼                      ▼
                    ┌──────────────┐         ┌─────────────────┐
                    │   Docker     │◀────────│  Generated      │
                    │  Container   │         │  Python Code    │
                    │  (Execution) │         │                 │
                    └──────────────┘         └─────────────────┘
                              │
                              ▼
                    ┌──────────────┐
                    │   Results    │
                    │ (JSON/Images)│
                    └──────────────┘
```

## Quick Start

### Prerequisites

- Python 3.11+
- Docker and Docker Compose
- Google Gemini API key

### Installation

1. **Clone the repository**
```bash
git clone <repository-url>
cd data-analysis-api
```

2. **Set up environment variables**
```bash
cp .env.example .env
# Edit .env with your Gemini API key
```

3. **Run with Docker Compose (Recommended)**
```bash
docker-compose up --build
```

4. **Or run locally**
```bash
# Install dependencies
pip install -r requirements.txt

# Option 1: Use the Python startup script (recommended)
python run_server.py

# Option 2: Use the shell script
./start_server.sh

# Option 3: Use uvicorn directly (development mode with auto-reload)
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000 --reload-exclude 'sandbox/*' --reload-exclude 'logs/*'

# Option 4: Use uvicorn directly (production mode without auto-reload)  
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

**Configuration**: Set `ENABLE_RELOAD=True` in `.env` to enable auto-reload for development.

### API Usage

The API will be available at `http://localhost:8000`

- **API Documentation**: `http://localhost:8000/docs`
- **Health Check**: `http://localhost:8000/health`

#### Upload and Analyze Files

```bash
curl -X POST "http://localhost:8000/api/analyze" \
  -F "files=@data.csv" \
  -F "question=What are the main trends in this data?" \
  -F "analysis_type=statistical"
```

#### Example Response

For successful analysis, the API returns the analysis results directly:

```json
{
  "edge_count": 7,
  "highest_degree_node": "Bob",
  "average_degree": 2.8,
  "density": 0.7,
  "shortest_path_alice_eve": 2.0,
  "network_graph": "iVBORw0KGgoAAAANSUhEUgAAAo...",
  "degree_histogram": "iVBORw0KGgoAAAANSUhEUgAAAo..."
}
```

**Image Fields**: All image fields in the response contain plain base64-encoded PNG data (no `data:image/png;base64,` prefix). To display these images in a web browser, you'll need to add the appropriate data URI prefix.
```

For failed analysis, the API returns error metadata:

```json
{
  "request_id": "uuid-here",
  "status": "error",
  "question": "What are the main trends in this data?",
  "analysis_type": "statistical", 
  "files_processed": 1,
  "execution_time": 5.2,
  "error": "Analysis execution failed",
  "stderr": "Error details...",
  "stdout": "Output details..."
  }
}
```

## Project Structure

```
/
├── app/
│   ├── __init__.py          # Package initialization
│   ├── main.py              # FastAPI application
│   ├── logger.py            # Logging utilities
│   ├── utils.py             # File processing utilities
│   ├── llm.py              # Gemini API integration
│   └── docker_runner.py     # Docker execution engine
├── sandbox/                 # Temporary execution directories
├── logs/                    # Application logs
├── examples/               # Example files for testing
├── requirements.txt        # Python dependencies
├── Dockerfile             # Container definition
├── docker-compose.yml     # Multi-service setup
├── .env.example           # Environment variables template
└── README.md             # This file
```

## Supported Analysis Types

### Statistical Analysis
- Descriptive statistics
- Correlation analysis
- Distribution plots
- Hypothesis testing

### Network Analysis
- Graph metrics (centrality, clustering)
- Community detection
- Network visualizations
- Path analysis

### Time Series Analysis
- Trend analysis
- Seasonality detection
- Forecasting
- Rolling statistics

### Machine Learning
- Clustering
- Classification
- Feature analysis
- Model evaluation

### General Analysis
- Data exploration
- Quality assessment
- Custom visualizations
- Pattern detection

## Chart and Visualization Standards

All charts and plots generated by the system follow these mandatory standards:

### Axis Labeling Requirements
- **X-axis labels**: Must include descriptive text indicating what the axis represents, including units when applicable
- **Y-axis labels**: Must include descriptive text indicating what the axis represents, including units when applicable
- **Example**: Instead of generic labels like "x" or "values", use descriptive labels like "Time (hours)", "Sales Revenue ($)", "Temperature (°C)", or "Number of Customers"

### Image Format
- **API Response**: All image fields contain plain base64-encoded PNG data without data URI prefix
- **Usage**: To display in web browsers, prepend `data:image/png;base64,` to the base64 string
- **Integration**: Direct base64 strings make it easier to integrate with various frontend frameworks and mobile applications

### Quality Standards
- Charts are optimized for readability with appropriate sizing and color schemes
- Legends are included when multiple data series are present
- Titles provide clear context about what the visualization represents

## Supported File Types

| Type | Extensions | Description |
|------|------------|-------------|
| CSV | `.csv`, `.tsv` | Comma/tab separated values |
| JSON | `.json` | JavaScript Object Notation |
| Excel | `.xlsx`, `.xls` | Microsoft Excel files |
| Text | `.txt`, `.log`, `.md` | Plain text files |
| HTML | `.html`, `.htm` | Web pages and tables |
| Archive | `.zip` | Compressed file collections |

## Security Features

- **Sandboxed Execution**: Code runs in isolated Docker containers
- **Resource Limits**: Memory and CPU constraints
- **Network Isolation**: No internet access during execution
- **Code Validation**: Static analysis of generated code
- **File Type Validation**: Restricted file upload types
- **Timeout Protection**: Prevents infinite loops

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `GEMINI_API_KEY` | - | Google Gemini API key (required) |
| `LOG_LEVEL` | `INFO` | Logging level |
| `DOCKER_TIMEOUT` | `300` | Max execution time (seconds) |
| `MAX_FILE_SIZE` | `50MB` | Maximum upload file size |
| `MAX_FILES_PER_REQUEST` | `10` | Maximum files per request |

### Docker Configuration

The system uses Docker to safely execute generated code:

- **Base Image**: `python:3.11-slim`
- **Memory Limit**: 512MB
- **CPU Limit**: 1 core
- **Network**: Disabled
- **User**: Non-root execution

## Development

### Running Tests

```bash
pytest tests/
```

### Code Quality

```bash
# Format code
black app/

# Type checking
mypy app/

# Linting
flake8 app/
```

### Adding New Analysis Types

1. Update `get_analysis_type_guidance()` in `llm.py`
2. Add specific prompting logic for the new type
3. Update documentation and examples
4. Ensure all chart generation follows axis labeling standards (descriptive labels with units)
5. Verify image output returns plain base64 strings without data URI prefix

## API Reference

### POST `/api/analyze`

Analyze uploaded files with natural language questions.

**Parameters:**
- `files`: List of files to upload (multipart/form-data)
- `question`: Analysis question (string)
- `analysis_type`: Type of analysis (optional, default: "general")
- `timeout`: Execution timeout in seconds (optional, default: 60)

**Response:**

For successful analysis, returns the analysis results directly:
```json
{
  "key1": "value1",
  "key2": 123,
  "chart_image": "iVBORw0KGgoAAAANSUhEUgAAAo..."
}
```

**Image Fields**: Image fields contain plain base64-encoded PNG data without the `data:image/png;base64,` prefix. All charts include properly labeled axes with descriptive text and units.

For failed analysis, returns error metadata:
```json
{
  "request_id": "string",
  "status": "error",
  "question": "string", 
  "analysis_type": "string",
  "files_processed": "integer",
  "execution_time": "float",
  "error": "string",
  "stderr": "string",
  "stdout": "string"
}
```

### GET `/health`

Health check endpoint.

**Response:**
```json
{
  "status": "healthy",
  "directories": {
    "sandbox": true,
    "logs": true,
    "examples": true
  }
}
```

## Troubleshooting

### Common Issues

1. **Docker Permission Errors**
   - Ensure Docker daemon is running
   - Check user permissions for Docker socket

2. **Gemini API Errors**
   - Verify API key is correct
   - Check API quota and limits

3. **Execution Timeouts**
   - Increase timeout parameter
   - Optimize data processing approach

4. **Memory Issues**
   - Reduce file sizes
   - Increase Docker memory limits

### Logs

Application logs are stored in the `logs/` directory:
- `api_YYYYMMDD.log`: General application logs
- `errors_YYYYMMDD.log`: Error-specific logs

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- [FastAPI](https://fastapi.tiangolo.com/) for the web framework
- [Google Generative AI](https://ai.google.dev/) for code generation
- [Docker](https://www.docker.com/) for secure execution
- [Pandas](https://pandas.pydata.org/) for data processing