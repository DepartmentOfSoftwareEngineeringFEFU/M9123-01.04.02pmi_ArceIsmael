#!/usr/bin/env python3
"""
Treatment Section Extraction Evaluator

This script evaluates the LangChainWrapper system that:
1. Detects treatment sections in medical documents using RAG
2. Extracts treatment section names and boundaries
3. Identifies subsections within treatments

Uses exact string matching for precise evaluation.
"""

import json
import time
from pathlib import Path
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import statistics
import re
import typer
from utils.llm_client import LangChainWrapper
from utils.pdf_processor import PDFProcessor, PDFCorruptionError
from utils.handlers import (
    extract_section_content,
    extract_text_between_quotes,
    extract_text_from_iacpaas_response,
    remove_trailing_dots,
    clean_subsections,
)
from utils.prompts import (
    rag_prompt,
    prompt_for_second_marker_extraction,
    template_for_subsection_extraction,
    iacpaas_subsection_prompt,
)


app = typer.Typer(
    help="Evaluate LangChainWrapper treatment section detection and subsection extraction"
)


@dataclass
class TreatmentExtractionMetrics:
    """Metrics for treatment section extraction evaluation"""

    model_name: str
    dataset_name: str
    total_samples: int

    # Treatment Detection Metrics
    treatment_detection_precision: float = 0.0
    treatment_detection_recall: float = 0.0
    treatment_detection_f1: float = 0.0
    treatment_detection_accuracy: float = 0.0

    # Treatment Name Extraction Metrics (Exact Match)
    treatment_name_exact_match_rate: float = 0.0
    treatment_name_precision: float = 0.0
    treatment_name_recall: float = 0.0
    treatment_name_f1: float = 0.0

    # Subsection Extraction Metrics (Exact Match)
    subsection_exact_match_rate: float = 0.0
    subsection_precision: float = 0.0
    subsection_recall: float = 0.0
    subsection_f1: float = 0.0

    # Overall Performance
    overall_accuracy: float = 0.0
    extraction_success_rate: float = 0.0

    # Processing Summary
    successful_extractions: int = 0
    failed_extractions: int = 0
    processing_errors: int = 0

    # Detailed Results
    sample_results: List[Dict] = field(default_factory=list)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


class TreatmentExtractionEvaluator:
    """
    Evaluator for LangChainWrapper treatment section detection and subsection extraction

    Evaluates:
    - Treatment section detection (binary classification)
    - Treatment name extraction (exact string matching)
    - Subsection identification (exact list matching)
    """

    def __init__(
        self,
        case_sensitive: bool = False,
        normalize_whitespace: bool = True,
    ):
        """
        Initialize the treatment extraction evaluator

        Args:
            case_sensitive: Whether to perform case-sensitive matching
            normalize_whitespace: Whether to normalize whitespace before matching
        """
        self.case_sensitive = case_sensitive
        self.normalize_whitespace = normalize_whitespace

    def _normalize_text(self, text: str) -> str:
        """Enhanced normalize text for comparison with better handling of minor differences"""
        if not text:
            return ""

        text = str(text).strip()

        text = text.rstrip(".,;")

        if self.normalize_whitespace:
            text = " ".join(text.split())

        if not self.case_sensitive:
            text = text.lower()

        return text

    def _exact_match(self, text1: str, text2: str) -> bool:
        """Check if two texts match exactly after normalization"""
        norm_text1 = self._normalize_text(text1)
        norm_text2 = self._normalize_text(text2)
        return norm_text1 == norm_text2

    def _exact_list_match(self, list1: List[str], list2: List[str]) -> bool:
        """Check if two lists have exact same elements (order independent)"""
        return set(self._normalize_text(item) for item in list1) == set(
            self._normalize_text(item) for item in list2
        )

    def extract_treatment_data_with_langchain(
        self, pdf_path: str, llm_wrapper: LangChainWrapper
    ) -> Dict[str, Any]:
        """
        Extract treatment data using LangChainWrapper

        Args:
            pdf_path: Path to PDF file
            llm_wrapper: LangChainWrapper instance

        Returns:
            Dictionary with extracted treatment data
        """
        try:
            try:
                pdf_processor = PDFProcessor(pdf_path)
                splits = pdf_processor.split_documents()
            except PDFCorruptionError as e:
                return {
                    "has_treatment": False,
                    "treatment_name": None,
                    "subsections": [],
                    "error": f"PDF_CORRUPTION: {str(e)}",
                }

            llm_wrapper.create_rag_chain(splits)

            results = llm_wrapper.query_rag_chain(rag_prompt)

            if "No treatment heading" in results["answer"]:
                return {
                    "has_treatment": False,
                    "treatment_name": None,
                    "subsections": [],
                    "error": None,
                }

            start_marker = extract_text_between_quotes(results["answer"])
            results = llm_wrapper.query_rag_chain(
                prompt_for_second_marker_extraction.format(start_marker=start_marker)
            )

            end_marker = extract_text_between_quotes(results["answer"])

            if start_marker is None:
                return {
                    "has_treatment": True,
                    "treatment_name": None,
                    "subsections": [],
                    "error": "Could not extract start marker",
                }

            if end_marker is None:
                end_marker = "END_OF_SECTION"

            try:
                full_text = pdf_processor.get_full_text()
            except PDFCorruptionError as e:
                return {
                    "has_treatment": True,
                    "treatment_name": start_marker,
                    "subsections": [],
                    "error": f"PDF_CORRUPTION: {str(e)}",
                }

            extracted_section = extract_section_content(
                full_text, start_marker, remove_trailing_dots(end_marker)
            )

            if extracted_section is None or not extracted_section.strip():
                return {
                    "has_treatment": True,
                    "treatment_name": start_marker,
                    "subsections": [],
                    "error": "Could not extract section content",
                }

            splits = pdf_processor.split_documents(
                separators=None,
                chunk_size=6000,
                chunk_overlap=200,
                are_docs=False,
                text=extracted_section,
            )

            possible_subsections = []
            last_found_subsection = None

            batch_size = 50
            for i, split in enumerate(splits[:batch_size]):
                previous_context = (
                    last_found_subsection
                    if last_found_subsection is not None
                    else "No previous subsection found"
                )

                answer_from_llm = llm_wrapper.query_llm(
                    template_for_subsection_extraction.format(
                        text=split, previous_context=previous_context
                    )
                )

                if "not found" not in answer_from_llm.content.lower():
                    possible_subsections.append(answer_from_llm.content)
                    last_found_subsection = answer_from_llm.content

                if i < len(splits[:batch_size]) - 1:
                    time.sleep(0.5)

            subsections = []
            for subsection in possible_subsections:
                if subsection and subsection.strip() and "not found" not in subsection.lower():
                    split_subsections = [s.strip() for s in subsection.split("\n") if s.strip()]
                    subsections.extend(split_subsections)

            clean_subsections_basic = [sub for sub in subsections if sub and sub.strip()]

            section_match = re.search(r"^(\d+)\.", start_marker.strip())
            main_section_number = section_match.group(1) if section_match else None

            final_clean_subsections = clean_subsections(
                clean_subsections_basic, main_section_number
            )

            return {
                "has_treatment": True,
                "treatment_name": start_marker,
                "subsections": final_clean_subsections,
                "error": None,
            }

        except Exception as e:
            typer.echo(f"Error extracting treatment data: {str(e)}")
            return {
                "has_treatment": False,
                "treatment_name": None,
                "subsections": [],
                "error": str(e),
            }

    def evaluate_treatment_extraction(
        self,
        validation_dataset: List[Dict],
        model_name: str = "langchain_wrapper",
        temperature: float = 0.0,
        max_samples: int = None,
    ) -> TreatmentExtractionMetrics:
        """
        Evaluate LangChainWrapper treatment extraction system

        Args:
            validation_dataset: List of validation samples with ground truth
            model_name: Name of the model being evaluated
            temperature: LLM temperature
            max_samples: Maximum number of samples to evaluate

        Returns:
            TreatmentExtractionMetrics object with evaluation results
        """
        if max_samples:
            validation_dataset = validation_dataset[:max_samples]

        typer.echo(f"Starting LangChainWrapper evaluation for {len(validation_dataset)} samples")

        llm_wrapper = LangChainWrapper(temperature=temperature)

        metrics = TreatmentExtractionMetrics(
            model_name=model_name,
            dataset_name="treatment_extraction",
            total_samples=len(validation_dataset),
        )

        sample_results = []
        for i, sample in enumerate(validation_dataset):
            typer.echo(f"Processing sample {i+1}/{len(validation_dataset)}")

            try:
                pdf_path = sample.get("pdf_path") or sample.get("source_file")
                ground_truth = sample.get("ground_truth") or sample.get("expected")

                if not pdf_path or not ground_truth:
                    typer.echo(f"Sample {i} missing required fields, skipping")
                    continue

                extracted_data = self.extract_treatment_data_with_langchain(pdf_path, llm_wrapper)

                sample_result = self._evaluate_single_sample(
                    extracted_data, ground_truth, pdf_path, i
                )
                sample_results.append(sample_result)

                time.sleep(3)

            except Exception as e:
                typer.echo(f"Error processing sample {i}: {str(e)}")
                metrics.processing_errors += 1
                sample_results.append({"sample_id": i, "error": str(e), "overall_success": False})

        self._compute_aggregate_metrics(metrics, sample_results)

        metrics.sample_results = sample_results

        return metrics

    def _evaluate_single_sample(
        self, extracted: Dict, truth: Dict, source: str, sample_id: int
    ) -> Dict:
        """Evaluate a single sample"""

        extracted_has_treatment = extracted.get("has_treatment", False)
        truth_has_treatment = truth.get("has_treatment", False)
        treatment_detection_correct = extracted_has_treatment == truth_has_treatment

        extracted_treatment_name = extracted.get("treatment_name")
        truth_treatment_name = truth.get("treatment_name")

        treatment_name_exact_match = self._exact_match(
            extracted_treatment_name or "", truth_treatment_name or ""
        )

        if treatment_name_exact_match:
            treatment_name_precision, treatment_name_recall = 1.0, 1.0
        elif not extracted_treatment_name and not truth_treatment_name:
            treatment_name_precision, treatment_name_recall = 1.0, 1.0
        else:
            treatment_name_precision, treatment_name_recall = 0.0, 0.0

        extracted_subsections = self._extract_subsections(extracted)
        truth_subsections = self._extract_subsections(truth)

        subsection_exact_match = self._exact_list_match(extracted_subsections, truth_subsections)
        subsection_precision, subsection_recall = self._calculate_precision_recall(
            extracted_subsections, truth_subsections
        )

        overall_success = (
            treatment_detection_correct and treatment_name_exact_match and subsection_exact_match
        )

        return {
            "sample_id": sample_id,
            "source": source,
            "extracted_has_treatment": extracted_has_treatment,
            "truth_has_treatment": truth_has_treatment,
            "treatment_detection_correct": treatment_detection_correct,
            "extracted_treatment_name": extracted_treatment_name,
            "truth_treatment_name": truth_treatment_name,
            "treatment_name_exact_match": treatment_name_exact_match,
            "treatment_name_precision": treatment_name_precision,
            "treatment_name_recall": treatment_name_recall,
            "extracted_subsections": extracted_subsections,
            "truth_subsections": truth_subsections,
            "subsection_exact_match": subsection_exact_match,
            "subsection_precision": subsection_precision,
            "subsection_recall": subsection_recall,
            "overall_success": overall_success,
            "extraction_error": extracted.get("error"),
        }

    def _extract_subsections(self, data: Dict) -> List[str]:
        """Extract subsection names from data structure"""
        subsections = []

        if "subsections" in data:
            subs = data["subsections"]
            if isinstance(subs, list):
                subsections.extend([str(sub).strip() for sub in subs if sub and str(sub).strip()])
            elif subs and str(subs).strip():
                subsections.append(str(subs).strip())

        if "treatment_subsections" in data:
            subs = data["treatment_subsections"]
            if isinstance(subs, list):
                subsections.extend([str(sub).strip() for sub in subs if sub and str(sub).strip()])

        if "sections" in data:
            sections = data["sections"]
            if isinstance(sections, list):
                for section in sections:
                    if isinstance(section, dict):
                        if "subsections" in section:
                            sub_subs = section["subsections"]
                            if isinstance(sub_subs, list):
                                subsections.extend(
                                    [
                                        str(sub).strip()
                                        for sub in sub_subs
                                        if sub and str(sub).strip()
                                    ]
                                )

        return subsections

    def _calculate_precision_recall(
        self, extracted: List[str], truth: List[str]
    ) -> Tuple[float, float]:
        """Calculate precision and recall for lists"""
        if not extracted and not truth:
            return 1.0, 1.0
        if not extracted:
            return 0.0, 0.0
        if not truth:
            return 0.0, 0.0

        norm_extracted = set([self._normalize_text(item) for item in extracted])
        norm_truth = set([self._normalize_text(item) for item in truth])

        intersection = norm_extracted.intersection(norm_truth)

        precision = len(intersection) / len(norm_extracted) if norm_extracted else 0.0
        recall = len(intersection) / len(norm_truth) if norm_truth else 0.0

        return precision, recall

    def _compute_aggregate_metrics(
        self, metrics: TreatmentExtractionMetrics, sample_results: List[Dict]
    ):
        """Compute aggregate metrics from sample results"""
        if not sample_results:
            return

        valid_results = [r for r in sample_results if not r.get("error")]
        metrics.successful_extractions = len(valid_results)
        metrics.failed_extractions = len(sample_results) - len(valid_results)

        if not valid_results:
            return

        detection_correct = [r["treatment_detection_correct"] for r in valid_results]
        metrics.treatment_detection_accuracy = sum(detection_correct) / len(detection_correct)

        true_positives = sum(
            1 for r in valid_results if r["extracted_has_treatment"] and r["truth_has_treatment"]
        )
        false_positives = sum(
            1
            for r in valid_results
            if r["extracted_has_treatment"] and not r["truth_has_treatment"]
        )
        false_negatives = sum(
            1
            for r in valid_results
            if not r["extracted_has_treatment"] and r["truth_has_treatment"]
        )

        metrics.treatment_detection_precision = (
            true_positives / (true_positives + false_positives)
            if (true_positives + false_positives) > 0
            else 0.0
        )
        metrics.treatment_detection_recall = (
            true_positives / (true_positives + false_negatives)
            if (true_positives + false_negatives) > 0
            else 0.0
        )
        metrics.treatment_detection_f1 = (
            2
            * (metrics.treatment_detection_precision * metrics.treatment_detection_recall)
            / (metrics.treatment_detection_precision + metrics.treatment_detection_recall)
            if (metrics.treatment_detection_precision + metrics.treatment_detection_recall) > 0
            else 0.0
        )

        name_exact_matches = [r["treatment_name_exact_match"] for r in valid_results]
        metrics.treatment_name_exact_match_rate = sum(name_exact_matches) / len(name_exact_matches)

        name_precisions = [r["treatment_name_precision"] for r in valid_results]
        name_recalls = [r["treatment_name_recall"] for r in valid_results]
        metrics.treatment_name_precision = statistics.mean(name_precisions)
        metrics.treatment_name_recall = statistics.mean(name_recalls)
        metrics.treatment_name_f1 = (
            2
            * (metrics.treatment_name_precision * metrics.treatment_name_recall)
            / (metrics.treatment_name_precision + metrics.treatment_name_recall)
            if (metrics.treatment_name_precision + metrics.treatment_name_recall) > 0
            else 0.0
        )

        subsection_exact_matches = [r["subsection_exact_match"] for r in valid_results]
        metrics.subsection_exact_match_rate = sum(subsection_exact_matches) / len(
            subsection_exact_matches
        )

        subsection_precisions = [r["subsection_precision"] for r in valid_results]
        subsection_recalls = [r["subsection_recall"] for r in valid_results]
        metrics.subsection_precision = statistics.mean(subsection_precisions)
        metrics.subsection_recall = statistics.mean(subsection_recalls)
        metrics.subsection_f1 = (
            2
            * (metrics.subsection_precision * metrics.subsection_recall)
            / (metrics.subsection_precision + metrics.subsection_recall)
            if (metrics.subsection_precision + metrics.subsection_recall) > 0
            else 0.0
        )

        overall_successes = [r["overall_success"] for r in valid_results]
        metrics.overall_accuracy = sum(overall_successes) / len(overall_successes)
        metrics.extraction_success_rate = metrics.successful_extractions / metrics.total_samples

    def debug_single_sample(self, pdf_path: str, ground_truth: Dict, sample_id: int = 0) -> Dict:
        """Debug a single sample with detailed output"""

        print(f"\n🔍 DEBUGGING LANGCHAIN TREATMENT EXTRACTION SAMPLE {sample_id}")
        print("=" * 60)

        llm_wrapper = LangChainWrapper(temperature=0.0)

        extracted_data = self.extract_treatment_data_with_langchain(pdf_path, llm_wrapper)

        result = self._evaluate_single_sample(extracted_data, ground_truth, pdf_path, sample_id)

        print(f"\nPDF Path: {pdf_path}")

        print(f"\nTreatment Detection:")
        print(f"  Extracted has treatment: {result['extracted_has_treatment']}")
        print(f"  Truth has treatment: {result['truth_has_treatment']}")
        print(f"  Detection correct: {result['treatment_detection_correct']}")

        print(f"\nTreatment Name:")
        print(f"  Extracted: {result['extracted_treatment_name']}")
        print(f"  Truth: {result['truth_treatment_name']}")
        print(f"  Exact match: {result['treatment_name_exact_match']}")
        print(f"  Precision: {result['treatment_name_precision']:.4f}")
        print(f"  Recall: {result['treatment_name_recall']:.4f}")

        print(f"\nSubsections:")
        print(f"  Extracted: {result['extracted_subsections']}")
        print(f"  Truth: {result['truth_subsections']}")
        print(f"  Exact match: {result['subsection_exact_match']}")
        print(f"  Precision: {result['subsection_precision']:.4f}")
        print(f"  Recall: {result['subsection_recall']:.4f}")

        print(f"\nOverall Success: {result['overall_success']}")

        if result.get("extraction_error"):
            print(f"\nExtraction Error: {result['extraction_error']}")

        return result

    def evaluate_treatment_extraction_openai(
        self,
        validation_dataset: List[Dict],
        model_name: str = "openai_model",
        openai_model: str = "gpt-4o-mini",
        temperature: float = 0.0,
        max_samples: int = None,
    ) -> TreatmentExtractionMetrics:
        """
        Evaluate treatment extraction using OpenAI models via LangChainWrapper

        Args:
            validation_dataset: List of validation samples with ground truth
            model_name: Name identifier for the model being evaluated
            openai_model: Specific OpenAI model name (e.g., "gpt-4o", "gpt-4o-mini")
            temperature: LLM temperature
            max_samples: Maximum number of samples to evaluate

        Returns:
            TreatmentExtractionMetrics object with evaluation results
        """
        if max_samples:
            validation_dataset = validation_dataset[:max_samples]

        typer.echo(
            f"Starting OpenAI evaluation with {openai_model} for {len(validation_dataset)} samples"
        )

        # Initialize LangChainWrapper with specific OpenAI model
        llm_wrapper = LangChainWrapper(engine=openai_model, temperature=temperature)

        # Initialize metrics
        metrics = TreatmentExtractionMetrics(
            model_name=f"{model_name}_{openai_model}",
            dataset_name="treatment_extraction",
            total_samples=len(validation_dataset),
        )

        # Process each sample
        sample_results = []
        for i, sample in enumerate(validation_dataset):
            typer.echo(f"Processing sample {i+1}/{len(validation_dataset)} with {openai_model}")

            try:
                # Extract required fields
                pdf_path = sample.get("pdf_path") or sample.get("source_file")
                ground_truth = sample.get("ground_truth") or sample.get("expected")

                if not pdf_path or not ground_truth:
                    typer.echo(f"Sample {i} missing required fields, skipping")
                    continue

                # Perform extraction using OpenAI model
                extracted_data = self.extract_treatment_data_with_langchain(pdf_path, llm_wrapper)

                # Evaluate single sample
                sample_result = self._evaluate_single_sample(
                    extracted_data, ground_truth, pdf_path, i
                )
                sample_results.append(sample_result)

                # Add delay to avoid rate limiting
                time.sleep(3)  # Increased delay between samples

            except Exception as e:
                typer.echo(f"Error processing sample {i} with {openai_model}: {str(e)}")
                metrics.processing_errors += 1
                sample_results.append({"sample_id": i, "error": str(e), "overall_success": False})

        self._compute_aggregate_metrics(metrics, sample_results)

        metrics.sample_results = sample_results

        return metrics


def load_validation_dataset(file_path: str) -> List[Dict[str, Any]]:
    """Load validation dataset from JSON file"""
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


@app.command()
def evaluate_langchain_rag(
    validation_dataset: Path = typer.Argument(..., help="Path to validation dataset JSON"),
    model_name: str = typer.Option("langchain_wrapper", help="Model identifier for results"),
    case_sensitive: bool = typer.Option(False, help="Use case-sensitive matching"),
    normalize_whitespace: bool = typer.Option(True, help="Normalize whitespace before matching"),
    temperature: float = typer.Option(0.0, help="LLM temperature"),
    max_samples: int = typer.Option(50, help="Maximum samples to evaluate"),
    output_file: Path = typer.Option(None, help="Output file for detailed results (JSON format)"),
):
    """
    Evaluate LangChainWrapper RAG
    """
    typer.echo(f"🔍 Evaluating LangChainWrapper RAG")
    typer.echo(f"📊 Dataset: {validation_dataset}")
    typer.echo(f"🤖 Model: {model_name}")
    typer.echo(f"🌡️  Temperature: {temperature}")
    typer.echo(f"📝 Max samples: {max_samples}")

    try:
        dataset = load_validation_dataset(str(validation_dataset))
        typer.echo(f"✅ Loaded {len(dataset)} samples from dataset")

        evaluator = TreatmentExtractionEvaluator(
            case_sensitive=case_sensitive,
            normalize_whitespace=normalize_whitespace,
        )

        typer.echo("🚀 Starting evaluation...")
        start_time = time.time()

        metrics = evaluator.evaluate_treatment_extraction(
            validation_dataset=dataset,
            model_name=model_name,
            temperature=temperature,
            max_samples=max_samples,
        )

        end_time = time.time()
        evaluation_time = end_time - start_time

        typer.echo(f"\n📊 EVALUATION RESULTS")
        typer.echo("=" * 50)
        typer.echo(f"Model: {metrics.model_name}")
        typer.echo(f"Total Samples: {metrics.total_samples}")
        typer.echo(f"Successful Extractions: {metrics.successful_extractions}")
        typer.echo(f"Failed Extractions: {metrics.failed_extractions}")
        typer.echo(f"Processing Errors: {metrics.processing_errors}")
        typer.echo(f"Evaluation Time: {evaluation_time:.2f} seconds")

        typer.echo(f"\n🎯 TREATMENT DETECTION METRICS")
        typer.echo(f"Accuracy: {metrics.treatment_detection_accuracy:.4f}")
        typer.echo(f"Precision: {metrics.treatment_detection_precision:.4f}")
        typer.echo(f"Recall: {metrics.treatment_detection_recall:.4f}")
        typer.echo(f"F1-Score: {metrics.treatment_detection_f1:.4f}")

        typer.echo(f"\n📝 TREATMENT NAME EXTRACTION METRICS")
        typer.echo(f"Exact Match Rate: {metrics.treatment_name_exact_match_rate:.4f}")
        typer.echo(f"Precision: {metrics.treatment_name_precision:.4f}")
        typer.echo(f"Recall: {metrics.treatment_name_recall:.4f}")
        typer.echo(f"F1-Score: {metrics.treatment_name_f1:.4f}")

        typer.echo(f"\n🔍 SUBSECTION EXTRACTION METRICS")
        typer.echo(f"Exact Match Rate: {metrics.subsection_exact_match_rate:.4f}")
        typer.echo(f"Precision: {metrics.subsection_precision:.4f}")
        typer.echo(f"Recall: {metrics.subsection_recall:.4f}")
        typer.echo(f"F1-Score: {metrics.subsection_f1:.4f}")

        typer.echo(f"\n🎯 OVERALL PERFORMANCE")
        typer.echo(f"Overall Accuracy: {metrics.overall_accuracy:.4f}")
        typer.echo(f"Extraction Success Rate: {metrics.extraction_success_rate:.4f}")

        if output_file:
            results_data = {
                "evaluation_metadata": {
                    "model_name": metrics.model_name,
                    "dataset_name": metrics.dataset_name,
                    "total_samples": metrics.total_samples,
                    "evaluation_time": evaluation_time,
                    "timestamp": metrics.timestamp,
                },
                "aggregate_metrics": {
                    "treatment_detection": {
                        "accuracy": metrics.treatment_detection_accuracy,
                        "precision": metrics.treatment_detection_precision,
                        "recall": metrics.treatment_detection_recall,
                        "f1_score": metrics.treatment_detection_f1,
                    },
                    "treatment_name_extraction": {
                        "exact_match_rate": metrics.treatment_name_exact_match_rate,
                        "precision": metrics.treatment_name_precision,
                        "recall": metrics.treatment_name_recall,
                        "f1_score": metrics.treatment_name_f1,
                    },
                    "subsection_extraction": {
                        "exact_match_rate": metrics.subsection_exact_match_rate,
                        "precision": metrics.subsection_precision,
                        "recall": metrics.subsection_recall,
                        "f1_score": metrics.subsection_f1,
                    },
                    "overall_performance": {
                        "overall_accuracy": metrics.overall_accuracy,
                        "extraction_success_rate": metrics.extraction_success_rate,
                    },
                },
                "processing_summary": {
                    "successful_extractions": metrics.successful_extractions,
                    "failed_extractions": metrics.failed_extractions,
                    "processing_errors": metrics.processing_errors,
                },
                "detailed_results": metrics.sample_results,
            }

            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(results_data, f, indent=2, ensure_ascii=False)

            typer.echo(f"📁 Detailed results exported to: {output_file}")
        typer.echo(f"\n✅ Evaluation completed successfully!")

    except Exception as e:
        typer.echo(f"❌ Error during evaluation: {str(e)}")
        raise typer.Exit(1)


@app.command()
def debug_sample(
    pdf_path: Path = typer.Argument(..., help="Path to PDF file"),
    ground_truth_file: Path = typer.Argument(..., help="Path to ground truth JSON"),
    sample_id: int = typer.Option(0, help="Sample ID for debugging"),
):
    """
    Debug a single sample with detailed output
    """
    try:
        with open(ground_truth_file, "r", encoding="utf-8") as f:
            ground_truth_data = json.load(f)

        if isinstance(ground_truth_data, list):
            if sample_id >= len(ground_truth_data):
                typer.echo(
                    f"❌ Sample ID {sample_id} out of range (max: {len(ground_truth_data)-1})"
                )
                raise typer.Exit(1)

            sample_data = ground_truth_data[sample_id]
            ground_truth = sample_data.get("ground_truth", {})

            if "pdf_path" in sample_data:
                pdf_path = Path(sample_data["pdf_path"])
        else:
            ground_truth = ground_truth_data

        evaluator = TreatmentExtractionEvaluator()

        result = evaluator.debug_single_sample(str(pdf_path), ground_truth, sample_id)

        return result

    except Exception as e:
        typer.echo(f"❌ Error debugging sample: {str(e)}")
        raise typer.Exit(1)


@app.command()
def evaluate_openai_models(
    validation_dataset: Path = typer.Argument(..., help="Path to validation dataset JSON"),
    models: str = typer.Option(
        "gpt-4o,gpt-4o-mini,gpt-3.5-turbo", help="Comma-separated list of OpenAI models to test"
    ),
    case_sensitive: bool = typer.Option(False, help="Use case-sensitive matching"),
    normalize_whitespace: bool = typer.Option(True, help="Normalize whitespace before matching"),
    temperature: float = typer.Option(0.0, help="LLM temperature"),
    max_samples: int = typer.Option(50, help="Maximum samples to evaluate"),
    output_dir: Path = typer.Option("./results", help="Directory to save output files"),
):
    """
    Evaluate multiple OpenAI models for treatment extraction and automatically save results
    """
    model_list = [model.strip() for model in models.split(",")]

    typer.echo(f"🔍 Evaluating OpenAI Models for Treatment Extraction")
    typer.echo(f"📊 Dataset: {validation_dataset}")
    typer.echo(f"🤖 Models: {', '.join(model_list)}")
    typer.echo(f"🌡️  Temperature: {temperature}")
    typer.echo(f"📝 Max samples: {max_samples}")
    typer.echo(f"📁 Output directory: {output_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    all_results = {}

    try:
        dataset = load_validation_dataset(str(validation_dataset))
        typer.echo(f"✅ Loaded {len(dataset)} samples from dataset")

        evaluator = TreatmentExtractionEvaluator(
            case_sensitive=case_sensitive,
            normalize_whitespace=normalize_whitespace,
        )

        for model_name in model_list:
            typer.echo(f"\n🚀 Starting evaluation for {model_name}...")
            start_time = time.time()

            try:
                metrics = evaluator.evaluate_treatment_extraction_openai(
                    validation_dataset=dataset,
                    model_name=model_name,
                    openai_model=model_name,
                    temperature=temperature,
                    max_samples=max_samples,
                )

                end_time = time.time()
                evaluation_time = end_time - start_time

                typer.echo(f"\n📊 RESULTS FOR {model_name.upper()}")
                typer.echo("=" * 60)
                typer.echo(f"Total Samples: {metrics.total_samples}")
                typer.echo(f"Successful Extractions: {metrics.successful_extractions}")
                typer.echo(f"Failed Extractions: {metrics.failed_extractions}")
                typer.echo(f"Processing Errors: {metrics.processing_errors}")
                typer.echo(f"Evaluation Time: {evaluation_time:.2f} seconds")

                typer.echo(f"\n🎯 TREATMENT DETECTION METRICS")
                typer.echo(f"Accuracy: {metrics.treatment_detection_accuracy:.4f}")
                typer.echo(f"Precision: {metrics.treatment_detection_precision:.4f}")
                typer.echo(f"Recall: {metrics.treatment_detection_recall:.4f}")
                typer.echo(f"F1-Score: {metrics.treatment_detection_f1:.4f}")

                typer.echo(f"\n🎯 OVERALL PERFORMANCE")
                typer.echo(f"Overall Accuracy: {metrics.overall_accuracy:.4f}")
                typer.echo(f"Extraction Success Rate: {metrics.extraction_success_rate:.4f}")

                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                safe_model_name = model_name.replace(".", "_").replace("-", "_")
                output_file = (
                    output_dir / f"treatment_extraction_results_{safe_model_name}_{timestamp}.json"
                )

                results_data = {
                    "evaluation_metadata": {
                        "model_name": metrics.model_name,
                        "openai_model": model_name,
                        "dataset_name": metrics.dataset_name,
                        "total_samples": metrics.total_samples,
                        "evaluation_time": evaluation_time,
                        "timestamp": metrics.timestamp,
                        "temperature": temperature,
                    },
                    "aggregate_metrics": {
                        "treatment_detection": {
                            "accuracy": metrics.treatment_detection_accuracy,
                            "precision": metrics.treatment_detection_precision,
                            "recall": metrics.treatment_detection_recall,
                            "f1_score": metrics.treatment_detection_f1,
                        },
                        "treatment_name_extraction": {
                            "exact_match_rate": metrics.treatment_name_exact_match_rate,
                            "precision": metrics.treatment_name_precision,
                            "recall": metrics.treatment_name_recall,
                            "f1_score": metrics.treatment_name_f1,
                        },
                        "subsection_extraction": {
                            "exact_match_rate": metrics.subsection_exact_match_rate,
                            "precision": metrics.subsection_precision,
                            "recall": metrics.subsection_recall,
                            "f1_score": metrics.subsection_f1,
                        },
                        "overall_performance": {
                            "overall_accuracy": metrics.overall_accuracy,
                            "extraction_success_rate": metrics.extraction_success_rate,
                        },
                    },
                    "processing_summary": {
                        "successful_extractions": metrics.successful_extractions,
                        "failed_extractions": metrics.failed_extractions,
                        "processing_errors": metrics.processing_errors,
                    },
                    "detailed_results": metrics.sample_results,
                }

                with open(output_file, "w", encoding="utf-8") as f:
                    json.dump(results_data, f, indent=2, ensure_ascii=False)

                typer.echo(f"📁 Results saved to: {output_file}")

                all_results[model_name] = {
                    "metrics": metrics,
                    "evaluation_time": evaluation_time,
                    "output_file": str(output_file),
                }

                typer.echo(f"✅ Evaluation completed for {model_name}!")

            except Exception as e:
                typer.echo(f"❌ Error evaluating {model_name}: {str(e)}")
                continue

        if len(all_results) > 1:
            typer.echo(f"\n📊 COMPARISON SUMMARY")
            typer.echo("=" * 80)
            typer.echo(
                f"{'Model':<20} {'Overall Acc':<12} {'Detection F1':<12} {'Name F1':<12} {'Subsection F1':<15} {'Time (s)':<10}"
            )
            typer.echo("-" * 80)

            for model_name, result in all_results.items():
                metrics = result["metrics"]
                eval_time = result["evaluation_time"]
                typer.echo(
                    f"{model_name:<20} {metrics.overall_accuracy:<12.4f} {metrics.treatment_detection_f1:<12.4f} {metrics.treatment_name_f1:<12.4f} {metrics.subsection_f1:<15.4f} {eval_time:<10.1f}"
                )

            comparison_file = (
                output_dir
                / f"model_comparison_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            )
            comparison_data = {
                "comparison_metadata": {
                    "timestamp": datetime.now().isoformat(),
                    "models_evaluated": list(all_results.keys()),
                    "dataset": str(validation_dataset),
                    "temperature": temperature,
                    "max_samples": max_samples,
                },
                "model_results": {
                    model_name: {
                        "overall_accuracy": result["metrics"].overall_accuracy,
                        "treatment_detection_f1": result["metrics"].treatment_detection_f1,
                        "treatment_name_f1": result["metrics"].treatment_name_f1,
                        "subsection_f1": result["metrics"].subsection_f1,
                        "evaluation_time": result["evaluation_time"],
                        "output_file": result["output_file"],
                    }
                    for model_name, result in all_results.items()
                },
            }

            with open(comparison_file, "w", encoding="utf-8") as f:
                json.dump(comparison_data, f, indent=2, ensure_ascii=False)

            typer.echo(f"\n📁 Comparison summary saved to: {comparison_file}")

        typer.echo(f"\n✅ All evaluations completed successfully!")

    except Exception as e:
        typer.echo(f"❌ Error during evaluation: {str(e)}")
        raise typer.Exit(1)


if __name__ == "__main__":
    app()


# Expected Dataset Format:
# [
#   {
#     "pdf_path": "path/to/document.pdf",
#     "ground_truth": {
#       "has_treatment": true,
#       "treatment_name": "3. Лечение",  # Single string, not a list
#       "subsections": ["3.1 Консервативное лечение", "3.2 Хирургическое лечение"]
#     }
#   }
# ]
