import gradio as gr
import requests
import json
import base64
from PIL import Image
import io
import os

# Configuration
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")

def analyze_data_gradio(files, question, analysis_type):
    """
    Gradio interface function for data analysis
    """
    if not files:
        return "Please upload at least one file.", None, None
    
    if not question.strip():
        return "Please enter a question about your data.", None, None
    
    try:
        # Prepare files for API request
        files_data = []
        for file in files:
            if file is not None:
                files_data.append(
                    ('files', (file.name, open(file.name, 'rb'), 'application/octet-stream'))
                )
        
        # Prepare request data
        data = {
            'question': question,
            'analysis_type': analysis_type or 'general'
        }
        
        # Make API request
        response = requests.post(
            f"{API_BASE_URL}/api/analyze",
            files=files_data,
            data=data,
            timeout=300  # 5 minutes timeout
        )
        
        # Close file handles
        for _, (_, file_handle, _) in files_data:
            file_handle.close()
        
        if response.status_code == 200:
            result = response.json()
            
            # Check if this is the new direct JSON format or old metadata format
            if "request_id" in result and "status" in result:
                # Old format with metadata wrapper
                if result.get("status") == "success" and "results" in result:
                    analysis_data = result["results"].get("json", {})
                    images_data = result["results"].get("images", [])
                else:
                    return f"Analysis failed: {result.get('error', 'Unknown error')}", None, None
            else:
                # New format - direct analysis results
                analysis_data = result
                images_data = []
                
                # Extract image data from base64 strings in the analysis results
                for key, value in analysis_data.items():
                    if isinstance(value, str) and len(value) > 100 and value.replace('+', '').replace('/', '').replace('=', '').isalnum():
                        # This looks like a base64 string (long alphanumeric string)
                        images_data.append({
                            "filename": f"{key}.png",
                            "data": value
                        })
            
            # Format JSON results for display
            json_results = json.dumps(analysis_data, indent=2)
            
            # Process first image if any
            if images_data:
                first_image = images_data[0]
                try:
                    image_data = base64.b64decode(first_image['data'])
                    image = Image.open(io.BytesIO(image_data))
                    
                    summary = f"""
**Analysis Complete!**

**Question:** {question}
**Analysis Type:** {analysis_type}
**Images Generated:** {len(images_data)}

✅ **Analysis completed successfully**
                    """
                    
                    return summary, json_results, image
                except Exception as e:
                    return f"Analysis completed but error processing image: {str(e)}", json_results, None
            else:
                summary = f"""
**Analysis Complete!**

**Question:** {question}
**Analysis Type:** {analysis_type}

✅ **Analysis completed successfully** (no visualizations generated)
                """
                return summary, json_results, None
        
        else:
            # Handle error response
            try:
                error_data = response.json()
                error_msg = error_data.get("error", f"HTTP {response.status_code}")
                return f"❌ **Analysis Failed:** {error_msg}", None, None
            except:
                return f"❌ **API Error:** HTTP {response.status_code} - {response.text}", None, None
    
    except Exception as e:
        return f"❌ **Error:** {str(e)}", None, None

# Create Gradio interface
def create_gradio_interface():
    with gr.Blocks(
        title="AI Data Analysis",
        theme=gr.themes.Soft(),
        css="""
        .gradio-container {
            max-width: 1200px;
            margin: auto;
        }
        """
    ) as interface:
        
        gr.Markdown("""
        # 📊 AI Data Analysis Tool
        
        Upload your data files and ask questions in natural language to get AI-powered analysis and visualizations.
        
        **Supported file types:** CSV, JSON, Excel, HTML, TXT, ZIP
        
        **New:** API now returns analysis results directly for cleaner integration!
        """)
        
        with gr.Row():
            with gr.Column(scale=1):
                # Input section
                gr.Markdown("### 📁 Upload Files")
                files_input = gr.File(
                    file_count="multiple",
                    label="Select your data files",
                    file_types=[".csv", ".json", ".xlsx", ".xls", ".html", ".txt", ".zip"]
                )
                
                gr.Markdown("### ❓ Your Question")
                question_input = gr.Textbox(
                    label="What would you like to know about your data?",
                    placeholder="e.g., What are the main trends in this dataset?",
                    lines=3
                )
                
                analysis_type_input = gr.Dropdown(
                    choices=["general", "statistical", "network", "time_series", "ml"],
                    value="general",
                    label="Analysis Type",
                    info="Choose the type of analysis you want"
                )
                
                analyze_button = gr.Button("🔍 Analyze Data", variant="primary", size="lg")
            
            with gr.Column(scale=2):
                # Output section
                gr.Markdown("### 📋 Analysis Results")
                
                summary_output = gr.Markdown(label="Summary")
                
                with gr.Tabs():
                    with gr.TabItem("📊 Visualization"):
                        image_output = gr.Image(label="Generated Chart", show_label=False)
                    
                    with gr.TabItem("📄 Detailed Results"):
                        json_output = gr.Code(
                            label="Analysis Results (JSON)",
                            language="json",
                            show_label=False
                        )
        
        # Event handler
        analyze_button.click(
            fn=analyze_data_gradio,
            inputs=[files_input, question_input, analysis_type_input],
            outputs=[summary_output, json_output, image_output]
        )
        
        # Examples
        gr.Markdown("""
        ### 💡 Example Questions
        
        - "What are the key statistics and trends in this dataset?"
        - "Show me the correlation between different variables"
        - "Create a visualization of the data distribution"
        - "What patterns can you find in this time series data?"
        - "Perform clustering analysis on this dataset"
        
        ### 🔄 API Response Format
        
        The API now returns analysis results directly:
        ```json
        {
          "edge_count": 7,
          "highest_degree_node": "Bob", 
          "network_graph": "data:image/png;base64,..."
        }
        ```
        """)
    
    return interface

# Create and launch interface
if __name__ == "__main__":
    # Import FastAPI app for direct integration
    import sys
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'app'))
    
    # Start the FastAPI server in background
    import threading
    import uvicorn
    from app.main import app
    
    def start_api_server():
        uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
    
    # Start API server in background thread
    api_thread = threading.Thread(target=start_api_server, daemon=True)
    api_thread.start()
    
    # Wait a moment for server to start
    import time
    time.sleep(3)
    
    # Create and launch Gradio interface
    interface = create_gradio_interface()
    interface.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False
    )
