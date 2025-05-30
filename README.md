# PDF Extractor System

A comprehensive system for extracting and analyzing information from PDF documents using RAG (Retrieval Augmented Generation) techniques with advanced text processing and JSON export capabilities.

## Overview

This system provides powerful PDF processing capabilities with intelligent text extraction, subsection analysis, and structured data export. It leverages multiple AI models and advanced text processing techniques to extract meaningful information from complex PDF documents, particularly optimized for medical and technical documentation.

## Features

- **PDF Processing**: Extract and analyze text from PDF documents with intelligent chunking
- **RAG Integration**: Retrieval Augmented Generation using LangChain and vector stores
- **Subsection Detection**: Automatic identification and extraction of document subsections
- **JSON Export**: Convert processed PDF data to structured JSON format
- **CLI Interface**: Command-line tool for batch processing and automation
- **Medical Text Analysis**: Specialized prompts for medical document processing
- **Multiple LLM Support**: Compatible with OpenAI, Ollama, and Hugging Face models

## Prerequisites

- Python 3.11 or higher
- UV package manager
- OpenAI API key (for OpenAI models only)

## Installation

1. Clone the repository:

```bash
git clone https://github.com/ismaArce/pdf-extractor-system.git
cd pdf-extractor-system
```

2. Install dependencies using UV:

```bash
uv sync
```

3. Create a `.env` file in the root directory and add an OpenAI API key:

```bash
OPENAI_API_KEY=your_openai_api_key_here
```

## Architecture

The system is organized into several key components:

### Core Components

- **`main.py`**: CLI application using Typer for PDF to JSON conversion
- **`utils/pdf_processor.py`**: PDFProcessor class for document loading and text extraction
- **`utils/llm_client.py`**: LLM integration with LangChainWrapper and LLMClient classes
- **`utils/handlers.py`**: Text processing utilities for section extraction and manipulation
- **`utils/prompts.py`**: Specialized prompts for medical and technical document analysis

### Key Dependencies

- **LangChain**: RAG implementation and LLM orchestration
- **PDFPlumber/PyMuPDF**: PDF text extraction and processing
- **Typer**: CLI interface and command handling
- **Chroma**: Vector database for document retrieval
- **Jupyter**: Interactive development and examples

## Usage

### Command Line Interface

The primary way to use the system is through the CLI:

```bash
# Basic PDF processing
python main.py convert path/to/document.pdf

# Process PDF and export to JSON
python main.py convert path/to/document.pdf --export

# Specify custom output path
python main.py convert path/to/document.pdf --export --output results.json
```

### CLI Options

- `pdf_path`: Path to the PDF file to process (required)
- `--export, -e`: Export processed data to JSON file
- `--output, -o`: Specify custom output path (defaults to PDF name with .json extension)

### Programmatic Usage

You can also use the system programmatically:

```python
from utils.pdf_processor import PDFProcessor
from utils.llm_client import LangChainWrapper

# Initialize components
pdf_processor = PDFProcessor("document.pdf")
llm_client = LangChainWrapper()

# Process document
splits = pdf_processor.split_documents()
llm_client.create_rag_chain(splits)
results = llm_client.query_rag_chain()
```

### Jupyter Notebook Examples

The project includes interactive examples in the `examples/` directory:

- `examples/simple_RAG.ipynb`: Basic RAG implementation demonstration

## Processing Workflow

1. **PDF Loading**: Load and extract text from PDF documents
2. **Section Detection**: Identify document structure and subsections
3. **Text Chunking**: Split text into manageable chunks for processing
4. **RAG Processing**: Create vector embeddings and perform retrieval
5. **Content Extraction**: Extract specific information using specialized prompts
6. **Structured Output**: Generate JSON with extracted recommendations and metadata

## Output Format

The system generates structured JSON output with the following format:

```json
{
  "pdf_name": "document_name",
  "nodes": [
    {
      "subsection": "Section Title",
      "subsection_text": "Full section content...",
      "general_condition": "Extracted conditions...",
      "recommendations": [
        {
          "recommendation": "Specific recommendation text",
          "condition": {
            "structured_data": "extracted_information"
          }
        }
      ]
    }
  ]
}
```

## Configuration

The system can be configured through environment variables:

- `OPENAI_API_KEY`: Your OpenAI API key for GPT models
- Additional model-specific configurations can be set in the respective client classes
