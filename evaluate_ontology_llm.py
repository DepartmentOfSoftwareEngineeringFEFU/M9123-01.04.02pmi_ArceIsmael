#!/usr/bin/env python3
"""
Evaluation Script for ontology extraction

This script provides:
- Semantic similarity using embeddings
- Statistical confidence intervals
- Text grounding verification
- Bootstrap sampling for robust statistics
"""

import json
import time
from pathlib import Path
from typing import List, Dict, Any

import typer
from utils.llm_client import LLMClient, OpenAIClient
from utils.prompts import adapter_system_prompt, adapter_user_prompt
from utils.ontology_evaluator import OntologyEvaluator

app = typer.Typer(help="Evaluation for medical extraction system")


def load_validation_dataset(file_path: str) -> List[Dict[str, Any]]:
    """Load validation dataset from JSON file"""
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def create_prompt_for_medical_extraction(text: str) -> str:
    """Create standardized prompt for medical extraction"""
    return f"""
Analyze the following medical text and extract medical recommendations in the specified JSON format.

IMPORTANT: Only extract information that is EXPLICITLY stated in the text. Do not infer or add any information not directly mentioned.

Text: {text}

Required JSON format:
{{
  "data": [
    {{
      "precursor": "medical treatment/intervention from text",
      "condition_group": {{
        "connection_between_blocks": "AND|OR",
        "condition_type": [
          {{
            "criteria": [
              {{
                "name": "parameter name from text",
                "criterion": {{
                  "value": "exact value from text",
                  "min_value": number,
                  "max_value": number,
                  "unit_of_measurement": "unit",
                  "condition": "additional condition description"
                }}
              }}
            ],
            "selection_rule": "ALL|ANY"
          }}
        ]
      }}
    }}
  ]
}}

Alternative format (when no condition_group needed):
{{
  "data": [
    {{
      "precursor": "medical treatment/intervention from text",
      "condition_type": [
        {{
          "criteria": [
            {{
              "name": "parameter name from text",
              "criterion": {{
                "value": "exact value from text",
                "min_value": number,
                "max_value": number,
                "unit_of_measurement": "unit",
                "condition": "additional condition description"
              }}
            }}
          ],
          "selection_rule": "ALL|ANY"
        }}
      ]
    }}
  ]
}}

IMPORTANT GUIDELINES:
1. "precursor" - Extract the exact medical treatment/intervention mentioned
2. Use "condition_group" when multiple condition blocks need AND/OR logic
3. Use direct "condition_type" when only one condition block is needed
4. In "criterion" object:
   - Use "value" for qualitative conditions (e.g., "присутствует", "положительный")
   - Use "min_value" for minimum thresholds (e.g., "выше 5")
   - Use "max_value" for maximum thresholds (e.g., "ниже 100")
   - Use both "min_value" and "max_value" for ranges (e.g., "от 60 до 90")
   - Include "unit_of_measurement" when specified (e.g., "мм рт. ст.", "ммоль/л")
   - Use "condition" for additional descriptive information
5. "selection_rule" can be "ALL" (требуется выполнение всех условий) or "ANY" (достаточно одного условия)

Extract ONLY what is explicitly stated. Do not infer or add any information not directly mentioned.
"""


@app.command()
def evaluate_ontology_model(
    validation_dataset: Path = typer.Argument(..., help="Path to validation dataset JSON"),
    model_name: str = typer.Option("iacpaas_llm", help="Model identifier for results"),
    max_samples: int = typer.Option(50, help="Maximum samples for evaluation (start small)"),
    temperature: float = typer.Option(0.01, help="LLM temperature"),
    confidence_level: float = typer.Option(0.95, help="Confidence level for intervals"),
    bootstrap_iterations: int = typer.Option(1000, help="Bootstrap iterations for CI"),
    semantic_model: str = typer.Option(
        "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
        help="Semantic similarity model",
    ),
    evaluating_trained_model: bool = typer.Option(
        False, help="this option will use the prompt for the trained model"
    ),
    llm_provider: str = typer.Option("iacpaas", help="LLM provider: 'iacpaas' or 'openai'"),
    openai_model: str = typer.Option(
        "gpt-4o-mini", help="OpenAI model to use when provider is 'openai'"
    ),
):

    typer.echo("🎓 Starting Medical Extraction Evaluation")
    typer.echo("=" * 80)

    if not validation_dataset.exists():
        typer.echo(f"❌ Validation dataset not found: {validation_dataset}")
        return

    # Load validation data
    typer.echo(f"📚 Loading validation dataset: {validation_dataset}")
    validation_data = load_validation_dataset(str(validation_dataset))
    typer.echo(f"   Total samples available: {len(validation_data)}")

    # Limit samples for evaluation
    eval_samples = validation_data[:max_samples]
    typer.echo(f"   Samples for evaluation: {len(eval_samples)}")

    # Initialize evaluator
    typer.echo(f"🔬 Initializing Evaluator...")
    typer.echo(f"   Semantic model: {semantic_model}")
    typer.echo(f"   Confidence level: {confidence_level*100}%")
    typer.echo(f"   Bootstrap iterations: {bootstrap_iterations}")

    evaluator = OntologyEvaluator(
        semantic_model=semantic_model,
        confidence_level=confidence_level,
        bootstrap_iterations=bootstrap_iterations,
    )

    # Initialize LLM client
    if llm_provider.lower() == "openai":
        llm_client = OpenAIClient(model=openai_model)
        typer.echo(f"🤖 OpenAI Client initialized (model: {openai_model})")
    elif llm_provider.lower() == "iacpaas":
        llm_client = LLMClient()
        typer.echo(f"🤖 IACPAAS LLM Client initialized")
    else:
        typer.echo(f"❌ Unknown LLM provider: {llm_provider}")
        return

    if llm_provider.lower() == "openai":
        model_name = f"{model_name}_openai_{openai_model.replace('-', '_')}"
    else:
        model_name = f"{model_name}_iacpaas"

    typer.echo(f"📊 Model identifier: {model_name}")

    typer.echo(f"\n🔄 Processing samples with {llm_provider.upper()} LLM...")

    extracted_data = []
    ground_truth = []
    source_texts = []
    successful_extractions = 0
    failed_extractions = 0

    for i, sample in enumerate(eval_samples):
        typer.echo(f"   Processing sample {i+1}/{len(eval_samples)}")

        source_text = sample["text"]
        expected_data = sample["data"]

        # Query LLM
        if evaluating_trained_model:
            prompt = adapter_user_prompt.format(text=source_text)
        else:
            prompt = create_prompt_for_medical_extraction(source_text)

        try:
            response = llm_client.query_llm(
                system_prompt=adapter_system_prompt,
                user_prompt=prompt,
                temperature=temperature,
                max_tokens=3000,
            )

            # Parse response
            clean_response = response.strip()
            if clean_response.startswith("```json"):
                clean_response = clean_response[7:]
            if clean_response.endswith("```"):
                clean_response = clean_response[:-3]

            try:
                extracted = json.loads(clean_response.strip())
            except json.JSONDecodeError as e:
                typer.echo(f"   ⚠️  JSON parsing failed for sample {i+1}: {str(e)}")
                typer.echo(f"   📤 Raw response: {clean_response[:300]}...")
                failed_extractions += 1
                continue  # Skip this sample

            # Handle case where LLM returns just the array (data value) instead of wrapped format
            if isinstance(extracted, list):
                # LLM returned just the array, wrap it in the expected format
                extracted = {"data": extracted}
            elif isinstance(extracted, dict) and "data" not in extracted:
                # LLM returned a single object, wrap it in data array
                extracted = {"data": [extracted]}

            # Store for evaluation
            extracted_data.append(extracted)
            ground_truth.append({"data": expected_data})
            source_texts.append(source_text)
            successful_extractions += 1

        except Exception as e:
            typer.echo(f"   ⚠️  Failed to process sample {i+1}: {e}")
            failed_extractions += 1
            continue

        time.sleep(0.1)

    typer.echo(f"\n📊 Processing Summary:")
    typer.echo(f"   Successful extractions: {successful_extractions}")
    typer.echo(f"   Failed extractions: {failed_extractions}")
    typer.echo(f"   Success rate: {successful_extractions/len(eval_samples)*100:.1f}%")

    if successful_extractions == 0:
        typer.echo("❌ No successful extractions to evaluate")
        return

    metrics = evaluator.evaluate_extraction_system(
        extracted_data=extracted_data,
        ground_truth=ground_truth,
        source_texts=source_texts,
        model_name=model_name,
    )

    # Print comprehensive academic report
    typer.echo(f"\n📋 ACADEMIC EVALUATION RESULTS:")
    evaluator.print_academic_report(metrics)

    # Additional insights
    typer.echo(f"\n💡 KEY INSIGHTS FOR THESIS:")

    # Statistical significance
    if metrics.semantic_similarity_ci_lower > 0.5:
        typer.echo("   ✅ Semantic similarity significantly above random baseline")
    else:
        typer.echo("   ❌ Semantic similarity not significantly above random baseline")

    typer.echo(f"\n📄 Results saved with detailed statistics for further analysis")


@app.command()
def test_ontology_single(
    validation_dataset: Path = typer.Argument(..., help="Path to validation dataset JSON"),
    sample_index: int = typer.Option(0, help="Index of sample to test"),
    temperature: float = typer.Option(0.01, help="LLM temperature"),
    evaluating_trained_model: bool = typer.Option(False, help="Evaluating trained model"),
    llm_provider: str = typer.Option("iacpaas", help="LLM provider: 'iacpaas' or 'openai'"),
    openai_model: str = typer.Option(
        "gpt-4o-mini", help="OpenAI model to use when provider is 'openai'"
    ),
):
    """
    Test ontology evaluation on a single sample for debugging
    """
    typer.echo(f"🧪 Testing Ontology Evaluation on Single Sample")

    if not validation_dataset.exists():
        typer.echo(f"❌ Validation dataset not found: {validation_dataset}")
        return

    validation_data = load_validation_dataset(str(validation_dataset))

    if sample_index >= len(validation_data):
        typer.echo(f"❌ Sample index {sample_index} out of range (max: {len(validation_data)-1})")
        return

    sample = validation_data[sample_index]

    typer.echo(f"📋 Sample {sample_index}:")
    typer.echo(f"📝 Source text: {sample['text'][:200]}...")
    typer.echo(f"🎯 Expected data: {len(sample['data'])} items")

    # Initialize evaluator
    evaluator = OntologyEvaluator()

    # Initialize LLM client
    if llm_provider.lower() == "openai":
        llm_client = OpenAIClient(model=openai_model)
        typer.echo(f"🤖 Using OpenAI: {openai_model}")
    else:
        llm_client = LLMClient()
        typer.echo(f"🤖 Using IACPAAS LLM")

    # Get LLM response
    if evaluating_trained_model:
        prompt = adapter_user_prompt.format(text=sample["text"])
    else:
        prompt = create_prompt_for_medical_extraction(sample["text"])

    try:
        response = llm_client.query_llm(
            system_prompt="You are a medical text analysis expert. Extract medical recommendations exactly as requested.",
            user_prompt=prompt,
            temperature=temperature,
            max_tokens=3000,
        )

        # Parse response
        clean_response = response.strip()
        if clean_response.startswith("```json"):
            clean_response = clean_response[7:]
        if clean_response.endswith("```"):
            clean_response = clean_response[:-3]

        try:
            extracted = json.loads(clean_response.strip())
        except json.JSONDecodeError as e:
            typer.echo(f"   ⚠️  JSON parsing failed for sample {sample_index}: {str(e)}")
            typer.echo(f"   📤 Raw response: {clean_response[:300]}...")
            return  # Exit on JSON parsing failure

        # Handle case where LLM returns just the array (data value) instead of wrapped format
        if isinstance(extracted, list):
            # LLM returned just the array, wrap it in the expected format
            extracted = {"data": extracted}
        elif isinstance(extracted, dict) and "data" not in extracted:
            # LLM returned a single object, wrap it in data array
            extracted = {"data": [extracted]}

        typer.echo(f"\n✅ LLM Response parsed successfully")
        typer.echo(f"📤 Extracted: {json.dumps(extracted, indent=2, ensure_ascii=False)}")

        # Run ontology evaluation on this single sample
        metrics = evaluator.evaluate_extraction_system(
            extracted_data=[extracted],
            ground_truth=[{"data": sample["data"]}],
            source_texts=[sample["text"]],
            model_name="single_test_ontology",
        )

        # Print detailed results
        evaluator.print_academic_report(metrics)

    except Exception as e:
        typer.echo(f"❌ JSON parsing failed: {e}")
        typer.echo(f"📤 Raw LLM Response: {response}")
        import traceback

        traceback.print_exc()
        return  # Exit completely on JSON failure


if __name__ == "__main__":
    app()
