"""
Medical Extraction Evaluator

This module provides statistically rigorous and semantically robust evaluation
for medical recommendation extraction systems.
"""

import json
import re
import statistics
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Tuple

import numpy as np
from scipy.stats import bootstrap
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import nltk
from nltk.corpus import stopwords
import pymorphy3

# Download required NLTK data
try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords")

# Russian stop words
RUSSIAN_STOP_WORDS = set(stopwords.words("russian"))

# Medical units and symbols to preserve during text cleaning
MEDICAL_SYMBOLS_PRESERVE = {
    "%",
    "°C",
    "°F",
    "ммHg",
    "мм рт. ст.",
    "кПа",
    "мг/дл",
    "ммоль/л",
    "мг/л",
    "г/л",
    "мл/мин",
    "уд/мин",
    "раз/мин",
    "мкг/мл",
    "нг/мл",
    "МЕ/л",
    "Ед/л",
    "г/сут",
    "мг/сут",
    "мкг/сут",
    "раза в день",
    "раз в день",
    "раза в сутки",
    "раз в сутки",
    "мг/кг",
    "мкг/кг",
}


MEDICAL_UNITS_PATTERN = (
    r"(\d+(?:[.,]\d+)?\s*(?:"
    + "|".join(re.escape(unit) for unit in MEDICAL_SYMBOLS_PRESERVE)
    + r"))"
)

MEDICAL_RANGES_PATTERN = r"(\d+(?:[.,]\d+)?[-/]\d+(?:[.,]\d+)?)"

COMPLEX_MEDICAL_PATTERN = r"(\d+(?:[.,]\d+)?(?:[-/]\d+(?:[.,]\d+)?)?\s*(?:°[CF]|ммhg|мм\s*рт\s*ст|мг/кг|ммоль/л|уд/мин|раз\s+в\s+(?:день|сутки)))"


@dataclass
class OntologyMetrics:
    """
    Comprehensive academic-grade metrics for medical extraction evaluation

    Includes confidence intervals, effect sizes, and statistical significance
    """

    model_name: str
    dataset_name: str
    total_samples: int

    # Semantic Similarity Metrics (with confidence intervals)
    semantic_similarity_mean: float = 0.0
    semantic_similarity_std: float = 0.0
    semantic_similarity_ci_lower: float = 0.0
    semantic_similarity_ci_upper: float = 0.0

    # Condition Extraction Metrics
    condition_precision: float = 0.0
    condition_recall: float = 0.0
    condition_f1: float = 0.0
    condition_precision_ci: Tuple[float, float] = (0.0, 0.0)
    condition_recall_ci: Tuple[float, float] = (0.0, 0.0)
    condition_f1_ci: Tuple[float, float] = (0.0, 0.0)

    tfidf_similarity_mean: float = 0.0
    tfidf_similarity_std: float = 0.0

    sample_results: List[Dict] = field(default_factory=list)

    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


class OntologyEvaluator:

    def __init__(
        self,
        semantic_model: str = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
        confidence_level: float = 0.95,
        bootstrap_iterations: int = 1000,
    ):
        """
        Initialize the evaluator

        Args:
            semantic_model: Sentence transformer model for semantic similarity
            confidence_level: Confidence level for intervals (default 95%)
            bootstrap_iterations: Number of bootstrap samples for CI estimation
        """
        self.confidence_level = confidence_level
        self.bootstrap_iterations = bootstrap_iterations

        try:
            self.semantic_model = SentenceTransformer(semantic_model)
        except Exception as e:
            raise RuntimeError(f"Semantic model initialization failed: {e}")

        self.results_history = []

        self._morph_analyzer = None
        self._morph_analyzer = pymorphy3.MorphAnalyzer()

    def _normalize_russian_text(self, text: str) -> str:
        """
        Normalize Russian text with lemmatization and medical symbol preservation
        """
        if not text or not isinstance(text, str):
            return ""

        medical_units = []
        normalized_text = text

        for match in re.finditer(COMPLEX_MEDICAL_PATTERN, normalized_text, re.IGNORECASE):
            unit = match.group(1)
            placeholder = f"__MEDICAL_UNIT_{len(medical_units)}__"
            medical_units.append(unit)
            normalized_text = normalized_text.replace(unit, placeholder, 1)

        for match in re.finditer(MEDICAL_RANGES_PATTERN, normalized_text):
            range_val = match.group(1)
            placeholder = f"__MEDICAL_UNIT_{len(medical_units)}__"
            medical_units.append(range_val)
            normalized_text = normalized_text.replace(range_val, placeholder, 1)

        for match in re.finditer(MEDICAL_UNITS_PATTERN, normalized_text, re.IGNORECASE):
            unit = match.group(1)
            placeholder = f"__MEDICAL_UNIT_{len(medical_units)}__"
            medical_units.append(unit)
            normalized_text = normalized_text.replace(unit, placeholder, 1)

        normalized_text = normalized_text.lower()

        normalized_text = re.sub(r"[^\w\s\d_]+", " ", normalized_text, flags=re.UNICODE)

        if self._morph_analyzer and self._is_russian_text(text):
            normalized_text = self._lemmatize_russian_text(normalized_text)

        normalized_text = re.sub(r"\s+", " ", normalized_text)

        for i, unit in enumerate(medical_units):
            placeholder = f"__MEDICAL_UNIT_{i}__"
            normalized_text = normalized_text.replace(placeholder, unit.lower())

        return normalized_text.strip()

    def _lemmatize_russian_text(self, text: str) -> str:
        """
        Lemmatize Russian text using pymorphy3
        """
        if not self._morph_analyzer:
            return text

        try:
            words = text.split()
            lemmatized_words = []

            for word in words:
                if "__MEDICAL_UNIT_" in word:
                    lemmatized_words.append(word)
                    continue

                clean_word = re.sub(r"[^\w\d]", "", word, flags=re.UNICODE)

                if clean_word and self._is_russian_text(clean_word):
                    parsed = self._morph_analyzer.parse(clean_word)[0]
                    lemma = parsed.normal_form

                    if lemma.lower() not in RUSSIAN_STOP_WORDS:
                        lemmatized_words.append(lemma)
                elif clean_word:
                    lemmatized_words.append(clean_word)

            return " ".join(lemmatized_words)

        except Exception as e:
            return text

    def _clean_medical_text(self, text: str) -> str:
        """
        Clean medical text while preserving important symbols and units

        Args:
            text: Input medical text

        Returns:
            Cleaned text with preserved medical symbols
        """
        if not text:
            return ""

        if self._is_russian_text(text):
            return self._normalize_russian_text(text)
        else:
            cleaned = re.sub(r"[^\w\s\d°%.,/-]", " ", text, flags=re.UNICODE)
            cleaned = re.sub(r"\s+", " ", cleaned)
            return cleaned.strip().lower()

    def _is_russian_text(self, text: str) -> bool:
        """Check if text is primarily Russian"""
        russian_chars = sum(1 for c in text if "\u0400" <= c <= "\u04ff")
        return russian_chars / len(text) > 0.5 if text else False

    def _calculate_tfidf_similarity(self, text1: str, text2: str) -> float:
        """Calculate TF-IDF similarity specifically for Russian text with normalization"""
        if not text1 or not text2:
            return 0.0

        # Apply normalization for Russian medical text
        normalized_text1 = self._clean_medical_text(text1)
        normalized_text2 = self._clean_medical_text(text2)

        if not normalized_text1 or not normalized_text2:
            return 0.0

        try:
            if len(normalized_text1.split()) < 2 or len(normalized_text2.split()) < 2:
                words1 = set(normalized_text1.split())
                words2 = set(normalized_text2.split())
                if not words1 or not words2:
                    return 0.0
                intersection = len(words1.intersection(words2))
                union = len(words1.union(words2))
                return intersection / union if union > 0 else 0.0

            vectorizer = TfidfVectorizer(
                lowercase=False,
                ngram_range=(1, 2),
                max_features=1000,
                stop_words=list(RUSSIAN_STOP_WORDS),
                token_pattern=r"(?u)\b\w+\b",
                min_df=1,
            )
            tfidf_matrix = vectorizer.fit_transform([normalized_text1, normalized_text2])
            similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
            return float(similarity)
        except Exception as e:
            words1 = set(normalized_text1.split())
            words2 = set(normalized_text2.split())
            if not words1 or not words2:
                return 0.0
            intersection = len(words1.intersection(words2))
            union = len(words1.union(words2))
            return intersection / union if union > 0 else 0.0

    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """Calculate text similarity using appropriate method based on language"""
        if not text1 or not text2:
            return 0.0

        normalized_text1 = self._clean_medical_text(text1)
        normalized_text2 = self._clean_medical_text(text2)

        if not normalized_text1 or not normalized_text2:
            return 0.0

        is_russian = self._is_russian_text(text1) or self._is_russian_text(text2)

        if is_russian:
            try:
                ext_emb = self.semantic_model.encode([normalized_text1])
                truth_emb = self.semantic_model.encode([normalized_text2])
                semantic_sim = cosine_similarity(ext_emb, truth_emb)[0][0]
                return float(semantic_sim)
            except Exception as e:
                words1 = set(normalized_text1.split())
                words2 = set(normalized_text2.split())
                if not words1 or not words2:
                    return 0.0
                intersection = len(words1.intersection(words2))
                union = len(words1.union(words2))
                return intersection / union if union > 0 else 0.0
        else:
            words1 = set(normalized_text1.split())
            words2 = set(normalized_text2.split())
            if not words1 or not words2:
                return 0.0
            intersection = len(words1.intersection(words2))
            union = len(words1.union(words2))
            return intersection / union if union > 0 else 0.0

    def evaluate_extraction_system(
        self,
        extracted_data: List[Dict],
        ground_truth: List[Dict],
        source_texts: List[str],
        model_name: str = "medical_llm",
    ) -> OntologyMetrics:

        if len(extracted_data) != len(ground_truth) or len(extracted_data) != len(source_texts):
            raise ValueError("Extracted data, ground truth, and source texts must have same length")

        metrics = OntologyMetrics(
            model_name=model_name,
            dataset_name="validation_dataset",
            total_samples=len(extracted_data),
        )

        sample_results = []

        for i, (extracted, truth, source) in enumerate(
            tqdm(zip(extracted_data, ground_truth, source_texts))
        ):
            sample_result = self._evaluate_single_sample(extracted, truth, source, i)
            sample_results.append(sample_result)

        self._compute_aggregate_metrics(metrics, sample_results)

        # self._compute_statistical_measures(metrics, sample_results)

        metrics.sample_results = sample_results

        self._save_results(metrics)

        return metrics

    def _evaluate_single_sample(
        self, extracted: Dict, truth: Dict, source: str, sample_id: int
    ) -> Dict:
        """Evaluate a single sample with comprehensive metrics"""
        result = {
            "sample_id": sample_id,
            "semantic_similarity": 0.0,
            "russian_tfidf_similarity": 0.0,
            "condition_match": False,
            "condition_precision": 0.0,
            "condition_recall": 0.0,
        }

        extracted_text = self._extract_text_content(extracted)
        truth_text = self._extract_text_content(truth)

        if extracted_text and truth_text:
            try:
                ext_emb = self.semantic_model.encode([extracted_text])
                truth_emb = self.semantic_model.encode([truth_text])
                semantic_sim = cosine_similarity(ext_emb, truth_emb)[0][0]
                result["semantic_similarity"] = float(semantic_sim)

                if self._is_russian_text(extracted_text) or self._is_russian_text(truth_text):
                    tfidf_sim = self._calculate_tfidf_similarity(extracted_text, truth_text)
                    result["russian_tfidf_similarity"] = float(tfidf_sim)

            except Exception as e:
                result["semantic_similarity"] = 0.0

        result["condition_match"], result["condition_precision"], result["condition_recall"] = (
            self._evaluate_condition_semantic(extracted, truth)
        )

        return result

    def _extract_text_content(self, data: Dict) -> str:
        """Extract all textual content from data structure for semantic comparison"""
        texts = []

        def extract_conditions_from_item(item):
            """Extract condition texts from a single item"""
            condition_texts = []
            if "condition_group" in item:
                condition_texts = self._extract_condition_texts({"data": [item]})
            elif "condition_type" in item:
                condition_texts = self._extract_condition_texts({"data": [item]})
            return condition_texts

        if isinstance(data, dict):
            if "data" in data:
                for item in data["data"]:
                    texts.extend(extract_conditions_from_item(item))

            if "recommendations" in data:
                for rec in data["recommendations"]:
                    if isinstance(rec, dict) and "recommendation" in rec:
                        texts.append(rec["recommendation"])

            if "condition_group" in data or "condition_type" in data:
                texts.extend(extract_conditions_from_item(data))

        return " ".join(filter(None, texts))

    def _evaluate_condition_semantic(
        self, extracted: Dict, truth: Dict
    ) -> Tuple[bool, float, float]:
        """Evaluate condition match using semantic similarity and return precision/recall"""
        extracted_conditions = self._extract_condition_texts(extracted)
        truth_conditions = self._extract_condition_texts(truth)

        if not extracted_conditions or not truth_conditions:
            return False, 0.0, 0.0

        true_positives = 0
        false_positives = 0
        false_negatives = 0
        matched_truth = set()

        for ext_cond in extracted_conditions:
            best_similarity = 0.0
            best_match_idx = -1

            for i, truth_cond in enumerate(truth_conditions):
                if i in matched_truth:
                    continue

                ext_emb = self.semantic_model.encode([ext_cond])
                truth_emb = self.semantic_model.encode([truth_cond])
                similarity = cosine_similarity(ext_emb, truth_emb)[0][0]

                if similarity > best_similarity:
                    best_similarity = similarity
                    best_match_idx = i

            if best_similarity > 0.6:
                true_positives += 1
                matched_truth.add(best_match_idx)
            else:
                false_positives += 1

        false_negatives = len(truth_conditions) - len(matched_truth)

        precision = (
            true_positives / (true_positives + false_positives)
            if (true_positives + false_positives) > 0
            else 0.0
        )
        recall = (
            true_positives / (true_positives + false_negatives)
            if (true_positives + false_negatives) > 0
            else 0.0
        )

        return len(matched_truth) > 0, precision, recall

    def _extract_condition_texts(self, data) -> List[str]:
        """Extract condition criterion texts for semantic comparison"""
        condition_texts = []

        def extract_from_criteria(criteria_list):
            for criterion in criteria_list:
                if isinstance(criterion, dict):
                    name = criterion.get("name", "")
                    crit_data = criterion.get("criterion", {})
                    if name:
                        text_parts = [str(name)]

                        if "value" in crit_data:
                            text_parts.append(str(crit_data["value"]))
                        if "min_value" in crit_data:
                            text_parts.append(f"min {str(crit_data['min_value'])}")
                            if "unit_of_measurement" in crit_data:
                                text_parts.append(str(crit_data["unit_of_measurement"]))
                        if "max_value" in crit_data:
                            text_parts.append(f"max {str(crit_data['max_value'])}")
                            if "unit_of_measurement" in crit_data:
                                text_parts.append(str(crit_data["unit_of_measurement"]))
                        if "condition" in crit_data:
                            text_parts.append(str(crit_data["condition"]))

                        text_parts = [part for part in text_parts if part and str(part).strip()]
                        if text_parts:
                            condition_text = " ".join(text_parts)
                            condition_texts.append(condition_text)

        def extract_from_condition_group(group):
            if isinstance(group, dict):
                if "condition_group" in group:
                    extract_from_condition_group(group["condition_group"])

                if "condition_type" in group:
                    for ctype in group["condition_type"]:
                        if "criteria" in ctype:
                            extract_from_criteria(ctype["criteria"])

        def process_item(item):
            if isinstance(item, dict):
                if "condition_group" in item:
                    extract_from_condition_group(item["condition_group"])
                elif "condition_type" in item:
                    extract_from_condition_group(item)

        if isinstance(data, dict):

            if "data" in data:
                for i, item in enumerate(data["data"]):
                    process_item(item)

            if "condition_group" in data or "condition_type" in data:
                process_item(data)

        elif isinstance(data, list):
            for i, item in enumerate(data):
                process_item(item)

        return condition_texts

    def _compute_aggregate_metrics(self, metrics: OntologyMetrics, sample_results: List[Dict]):
        """Compute aggregate metrics with confidence intervals"""

        semantic_similarities = [r["semantic_similarity"] for r in sample_results]
        if semantic_similarities:
            metrics.semantic_similarity_mean = statistics.mean(semantic_similarities)
            metrics.semantic_similarity_std = (
                statistics.stdev(semantic_similarities) if len(semantic_similarities) > 1 else 0.0
            )

            ci_lower, ci_upper = self._bootstrap_confidence_interval(semantic_similarities)
            metrics.semantic_similarity_ci_lower = ci_lower
            metrics.semantic_similarity_ci_upper = ci_upper

        condition_precisions = [r["condition_precision"] for r in sample_results]
        condition_recalls = [r["condition_recall"] for r in sample_results]

        if condition_precisions and condition_recalls:
            metrics.condition_precision = statistics.mean(condition_precisions)
            metrics.condition_recall = statistics.mean(condition_recalls)
            metrics.condition_f1 = (
                2
                * (metrics.condition_precision * metrics.condition_recall)
                / (metrics.condition_precision + metrics.condition_recall)
                if (metrics.condition_precision + metrics.condition_recall) > 0
                else 0.0
            )

            metrics.condition_precision_ci = self._bootstrap_confidence_interval(
                condition_precisions
            )
            metrics.condition_recall_ci = self._bootstrap_confidence_interval(condition_recalls)
            metrics.condition_f1_ci = self._bootstrap_confidence_interval(
                [
                    2 * (p * r) / (p + r) if (p + r) > 0 else 0.0
                    for p, r in zip(condition_precisions, condition_recalls)
                ]
            )

        tfidf_scores = [
            r["russian_tfidf_similarity"] for r in sample_results if "russian_tfidf_similarity" in r
        ]

        if tfidf_scores:
            metrics.tfidf_similarity_mean = statistics.mean(tfidf_scores)
            metrics.tfidf_similarity_std = (
                statistics.stdev(tfidf_scores) if len(tfidf_scores) > 1 else 0.0
            )
        else:
            # If no Russian text detected, use semantic similarity as fallback, for future works
            semantic_similarities = [r["semantic_similarity"] for r in sample_results]
            if semantic_similarities:
                metrics.tfidf_similarity_mean = statistics.mean(semantic_similarities)
                metrics.tfidf_similarity_std = (
                    statistics.stdev(semantic_similarities)
                    if len(semantic_similarities) > 1
                    else 0.0
                )

    def _bootstrap_confidence_interval(self, data: List[float]) -> Tuple[float, float]:
        """Calculate bootstrap confidence interval"""
        if len(data) <= 1:
            value = data[0] if data else 0.0
            return (value, value)

        if len(data) < 10:
            sorted_data = sorted(data)
            n = len(sorted_data)
            alpha = 1 - self.confidence_level
            lower_idx = max(0, int((alpha / 2) * n))
            upper_idx = min(n - 1, int((1 - alpha / 2) * n))
            return (sorted_data[lower_idx], sorted_data[upper_idx])

        try:

            def mean_statistic(x):
                return np.mean(x)

            res = bootstrap(
                (np.array(data),),
                mean_statistic,
                n_resamples=self.bootstrap_iterations,
                confidence_level=self.confidence_level,
                random_state=42,
            )
            return (res.confidence_interval.low, res.confidence_interval.high)
        except Exception as e:
            sorted_data = sorted(data)
            n = len(sorted_data)
            alpha = 1 - self.confidence_level
            lower_idx = max(0, int((alpha / 2) * n))
            upper_idx = min(n - 1, int((1 - alpha / 2) * n))
            return (sorted_data[lower_idx], sorted_data[upper_idx])

    def _convert_to_json_serializable(self, obj):
        """Convert numpy types and other non-serializable objects to JSON-serializable types"""
        if isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: self._convert_to_json_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_to_json_serializable(item) for item in obj]
        elif isinstance(obj, tuple):
            return tuple(self._convert_to_json_serializable(item) for item in obj)
        else:
            return obj

    def _save_results(self, metrics: OntologyMetrics):
        """Save evaluation results to file"""
        results_file = f"academic_evaluation_results_{metrics.model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

        # Convert metrics to dictionary for JSON serialization
        results_dict = {
            "model_name": metrics.model_name,
            "dataset_name": metrics.dataset_name,
            "total_samples": metrics.total_samples,
            "semantic_similarity": {
                "mean": float(metrics.semantic_similarity_mean),
                "std": float(metrics.semantic_similarity_std),
                "confidence_interval": [
                    float(metrics.semantic_similarity_ci_lower),
                    float(metrics.semantic_similarity_ci_upper),
                ],
            },
            "condition_metrics": {
                "precision": float(metrics.condition_precision),
                "recall": float(metrics.condition_recall),
                "f1": float(metrics.condition_f1),
                "precision_ci": [
                    float(metrics.condition_precision_ci[0]),
                    float(metrics.condition_precision_ci[1]),
                ],
                "recall_ci": [
                    float(metrics.condition_recall_ci[0]),
                    float(metrics.condition_recall_ci[1]),
                ],
                "f1_ci": [float(metrics.condition_f1_ci[0]), float(metrics.condition_f1_ci[1])],
            },
            "russian_similarity_scores": {
                "tfidf_cosine_similarity_mean": float(metrics.tfidf_similarity_mean),
                "tfidf_cosine_similarity_std": float(metrics.tfidf_similarity_std),
                "semantic_similarity_mean": float(metrics.semantic_similarity_mean),
            },
            "timestamp": metrics.timestamp,
            "sample_results": self._convert_to_json_serializable(metrics.sample_results),
        }

        with open(results_file, "w", encoding="utf-8") as f:
            json.dump(results_dict, f, indent=2, ensure_ascii=False)

    def print_academic_report(self, metrics: OntologyMetrics):
        """Print comprehensive academic evaluation report"""
        print("\n" + "=" * 80)
        print(f"📊 ONTOLOGY EXTRACTION EVALUATION REPORT")
        print("=" * 80)
        print(f"Model: {metrics.model_name}")
        print(f"Dataset: {metrics.dataset_name}")
        print(f"Total Samples: {metrics.total_samples}")
        print(f"Evaluation Date: {metrics.timestamp}")

        print(f"\n🔍 SEMANTIC SIMILARITY ANALYSIS")
        print("-" * 40)
        print(f"Mean Semantic Similarity (Transformers): {metrics.semantic_similarity_mean:.4f}")
        print(f"Standard Deviation: {metrics.semantic_similarity_std:.4f}")
        print(
            f"95% Confidence Interval: [{metrics.semantic_similarity_ci_lower:.4f}, {metrics.semantic_similarity_ci_upper:.4f}]"
        )

        print(f"\n🏥 CONDITION EXTRACTION METRICS")
        print("-" * 40)
        print(
            f"Precision: {metrics.condition_precision:.4f} (95% CI: {metrics.condition_precision_ci})"
        )
        print(f"Recall: {metrics.condition_recall:.4f} (95% CI: {metrics.condition_recall_ci})")
        print(f"F1-Score: {metrics.condition_f1:.4f} (95% CI: {metrics.condition_f1_ci})")

        print(f"\n📈 STATISTICAL ANALYSIS")
        print("-" * 40)

        print(f"\n📊 TEXT SIMILARITY METRICS")
        print("-" * 40)
        print(
            f"TF-IDF Similarity (Russian): {metrics.tfidf_similarity_mean:.4f} ± {metrics.tfidf_similarity_std:.4f}"
        )
        print(f"Note: Semantic similarity uses sentence transformers, TF-IDF used for Russian text")

        if metrics.semantic_similarity_mean > 0.7:
            print("✅ Strong semantic similarity - good content matching")
        elif metrics.semantic_similarity_mean > 0.4:
            print("⚠️  Moderate semantic similarity - some content matching issues")
        else:
            print("❌ Weak semantic similarity - significant content matching problems")
