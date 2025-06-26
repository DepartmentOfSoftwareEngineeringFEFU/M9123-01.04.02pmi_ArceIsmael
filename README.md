# Medical Treatment Recommendation Extraction System

A specialized academic research system for extracting and analyzing medical treatment recommendations from PDF documents using RAG (Retrieval Augmented Generation) techniques with statistically rigorous evaluation frameworks and structured JSON export capabilities.

## Overview

This system provides advanced medical document analysis capabilities specifically designed for extracting treatment recommendations from Russian medical clinical guidelines. It leverages multiple AI models, sophisticated text processing techniques, and comprehensive evaluation frameworks to extract structured medical information with statistical confidence intervals. The system is optimized for academic research in medical informatics, featuring ontological data extraction, treatment section detection, and robust evaluation methodologies for medical recommendation systems.

## Features

- **PDF Processing**: Extract and analyze text from PDF documents with intelligent chunking
- **RAG Integration**: Retrieval Augmented Generation using LangChain and vector stores
- **Subsection Detection**: Automatic identification and extraction of document subsections
- **JSON Export**: Convert processed PDF data to structured JSON format
- **CLI Interface**: Command-line tool for batch processing and automation
- **Medical Text Analysis**: Specialized prompts for medical document processing
- **Multiple LLM Support**: Compatible with OpenAI, Ollama, and Hugging Face models
- **Evaluation System**: Comprehensive evaluation scripts for ontology and treatment extraction with statistical analysis
- **PDF Corruption Detection**: Enhanced error handling for corrupted or invalid PDF files
- **Semantic Similarity Analysis**: Advanced evaluation using multilingual embeddings and confidence intervals

## Prerequisites

- Python 3.11 or higher
- UV package manager
- OpenAI API key (for OpenAI models only)
- IACPAAS API credentials (for IACPAAS models)

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

3. Create a `.env` file in the root directory and add required API keys:

```bash
OPENAI_API_KEY=your_openai_api_key_here
IACPAAS_TOKEN=your_iacpaas_token_here
IACPAAS_MODEL=your_iacpaas_model_name
```

## Architecture

The system is organized into several key components:

### Core Components

- **`main.py`**: CLI application using Typer for PDF to JSON conversion
- **`utils/pdf_processor.py`**: PDFProcessor class for document loading and text extraction with corruption detection
- **`utils/llm_client.py`**: LLM integration with LangChainWrapper and LLMClient classes
- **`utils/handlers.py`**: Text processing utilities for section extraction and manipulation
- **`utils/prompts.py`**: Specialized prompts for medical and technical document analysis
- **`utils/ontology_evaluator.py`**: Statistical evaluation system for medical recommendation extraction
- **`evaluate_ontology_llm.py`**: Evaluation script for ontology extraction
- **`evaluate_treatment_extraction.py`**: Comprehensive evaluation system for treatment section detection

### Key Dependencies

- **LangChain**: RAG implementation and LLM orchestration
- **PDFPlumber/PyMuPDF**: PDF text extraction and processing
- **Typer**: CLI interface and command handling
- **Chroma**: Vector database for document retrieval
- **Jupyter**: Interactive development and examples
- **NLP Libraries**:
  - `sentence-transformers`: Semantic similarity analysis
  - `transformers`: Multilingual BERT models for Russian text
  - `spacy`: Advanced NLP with Russian language support
  - `nltk`: Natural language processing
  - `pymorphy3`: Russian morphological analysis
- **Statistical Analysis**:
  - `scikit-learn`: Machine learning metrics and statistical tests
  - `scipy`: Statistical significance testing
  - `statsmodels`: Advanced statistical modeling
  - `pingouin`: Statistical functions with effect sizes
  - `evaluate`: Hugging Face evaluation library

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

## Evaluation Framework

The system includes a comprehensive evaluation framework for assessing the performance of medical text extraction. The framework supports two main evaluation types: ontology extraction and treatment section detection.

### Dataset Preparation

Before running evaluations, prepare your validation dataset in JSON format:

#### Ontology Evaluation Dataset Format

```json
[
  {
    "text": "Medical text containing recommendations...",
    "data": [
      {
        "precursor": "medical treatment/intervention",
        "condition_type": [
          {
            "criteria": [
              {
                "name": "parameter name",
                "criterion": {
                  "value": "exact value",
                  "min_value": 5,
                  "max_value": 10,
                  "unit_of_measurement": "mg",
                  "condition": "additional condition"
                }
              }
            ],
            "selection_rule": "ALL"
          }
        ]
      }
    ]
  }
]
```

#### Treatment Extraction Dataset Format

```json
[
  {
    "pdf_path": "path/to/document.pdf",
    "ground_truth": {
      "has_treatment": true,
      "treatment_name": "5. Лечение",
      "subsections": ["5.1 Консервативное лечение", "5.2 Хирургическое лечение"]
    }
  }
]
```

### Running Ontology Evaluation

#### Basic Evaluation

```bash
# Evaluate with IACPAAS model
python evaluate_ontology_llm.py evaluate-ontology-model data/validation_50.json

# Evaluate with OpenAI model
python evaluate_ontology_llm.py evaluate-ontology-model data/validation_50.json \
  --llm-provider openai \
  --openai-model gpt-4o-mini
```

#### Advanced Configuration

```bash
# Custom evaluation with specific parameters
python evaluate_ontology_llm.py evaluate-ontology-model data/validation_50.json \
  --model-name my_experiment \
  --max-samples 100 \
  --temperature 0.01 \
  --confidence-level 0.95 \
  --bootstrap-iterations 1000 \
  --semantic-model "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
```

#### Single Sample Testing

```bash
# Test specific sample for debugging
python evaluate_ontology_llm.py test-ontology-single data/validation_50.json \
  --sample-index 0 \
  --llm-provider openai \
  --openai-model gpt-4o-mini
```

### Running Treatment Extraction Evaluation

#### LangChain Wrapper Evaluation

```bash
# Basic evaluation with default LangChain wrapper
python evaluate_treatment_extraction.py evaluate-langchain data/treatment_dataset.json

# Custom parameters
python evaluate_treatment_extraction.py evaluate-langchain data/treatment_dataset.json \
  --max-samples 25 \
  --temperature 0.0 \
  --case-sensitive
```

#### Multi-Model OpenAI Evaluation

```bash
# Compare multiple OpenAI models
python evaluate_treatment_extraction.py evaluate-openai-models data/treatment_dataset.json \
  --models "gpt-4o,gpt-4o-mini,gpt-3.5-turbo" \
  --max-samples 50 \
  --output-dir ./results
```

#### Debugging Individual Samples

```bash
# Debug specific sample
python evaluate_treatment_extraction.py debug-single data/treatment_dataset.json \
  --sample-index 0

# Debug with specific PDF
python evaluate_treatment_extraction.py debug-single-pdf path/to/document.pdf \
  --expected-treatment "5. Лечение" \
  --expected-subsections "5.1 Консервативное лечение,5.2 Хирургическое лечение"
```

### Understanding Evaluation Results

#### Ontology Evaluation Metrics

The ontology evaluator provides comprehensive statistical analysis:

- **Semantic Similarity**: Cosine similarity using multilingual embeddings
- **Condition Extraction**: Precision, recall, and F1-score for condition matching
- **Statistical Confidence**: Bootstrap confidence intervals for robust statistics

Example output:

```
📊 ONTOLOGY EVALUATION RESULTS
═══════════════════════════════════════
Model: gpt-4o-mini_openai
Samples: 50/50 successful

🎯 SEMANTIC SIMILARITY
Mean: 0.756 (95% CI: 0.721-0.791)
Median: 0.782
Standard Deviation: 0.134

🔍 CONDITION EXTRACTION
Precision: 0.847 ± 0.023
Recall: 0.789 ± 0.031
F1-Score: 0.817 ± 0.025

📋 TEXT GROUNDING
Grounding Rate: 94.2%
```

#### Treatment Extraction Metrics

The treatment evaluator focuses on section detection and extraction:

- **Treatment Detection**: Binary classification accuracy
- **Name Extraction**: Exact match rate for treatment section names
- **Subsection Extraction**: Precision/recall for subsection identification
- **Overall Performance**: Combined accuracy across all tasks

Example output:

```
📊 TREATMENT EXTRACTION RESULTS
════════════════════════════════════════
Model: langchain_gpt4o
Total Samples: 25

🎯 TREATMENT DETECTION
Accuracy: 0.920
Precision: 0.958
Recall: 0.884
F1-Score: 0.920

📝 TREATMENT NAME EXTRACTION
Exact Match Rate: 0.840
Precision: 0.913
Recall: 0.913

🔍 SUBSECTION EXTRACTION
Exact Match Rate: 0.760
Precision: 0.856
Recall: 0.798

📊 OVERALL PERFORMANCE
Overall Accuracy: 0.720
Success Rate: 0.960
```

### Evaluation Best Practices

#### Dataset Quality

- Ensure ground truth annotations are accurate and consistent
- Include diverse examples covering different document types
- Validate PDF accessibility and text extraction quality

#### Statistical Significance

- Use adequate sample sizes (recommended: 50+ samples)
- Set appropriate confidence levels (default: 95%)
- Consider bootstrap iterations for robust statistics (1000+ recommended)

#### Model Comparison

- Use identical datasets across different models
- Control for temperature and other hyperparameters
- Document evaluation conditions for reproducibility

#### Error Analysis

- Review failed extractions for pattern identification
- Check PDF corruption issues using built-in detection
- Analyze semantic similarity distributions for insights

### Interpreting Confidence Intervals

The evaluation framework provides statistical confidence intervals:

- **Narrow intervals** (±0.01-0.03): High confidence in performance estimate
- **Wide intervals** (±0.05+): Consider increasing sample size or checking data quality
- **Non-overlapping intervals**: Statistically significant difference between models

### Troubleshooting Evaluation Issues

#### Common Problems

1. **PDF Corruption**: Use built-in corruption detection
2. **API Rate Limits**: Adjust delays between requests
3. **Memory Issues**: Reduce batch sizes or sample counts
4. **JSON Parsing Errors**: Enable debug mode for detailed error analysis

## Processing Workflow

1. **PDF Loading**: Load and extract text from PDF documents with corruption detection
2. **Section Detection**: Identify document structure and subsections
3. **Text Chunking**: Split text into manageable chunks for processing
4. **RAG Processing**: Create vector embeddings and perform retrieval
5. **Content Extraction**: Extract specific information using specialized prompts
6. **Structured Output**: Generate JSON with extracted recommendations and metadata
7. **Evaluation**: Statistical analysis and performance metrics (optional)

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
- `IACPAAS_TOKEN`: API token for IACPAAS models
- `IACPAAS_MODEL`: Model name for IACPAAS (e.g., "llama-3.3-70b")
- Additional model-specific configurations can be set in the respective client classes

## Error Handling

The system includes robust error handling for:

- **PDF Corruption**: Automatic detection and graceful handling of corrupted PDF files
- **Network Issues**: Retry logic for API calls with exponential backoff
- **JSON Parsing**: Validation and cleaning of LLM responses
- **Missing Dependencies**: Clear error messages for missing requirements

## Academic Research Features

This system is designed for academic research with:

- **Reproducible Results**: Fixed random seeds and deterministic evaluation
- **Statistical Significance**: Confidence intervals and significance testing
- **Detailed Logging**: Comprehensive logs for debugging and analysis
- **Benchmark Datasets**: Support for standardized evaluation datasets
- **Academic Metrics**: Standard NLP evaluation metrics and custom medical domain metrics
