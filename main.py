import typer
from pathlib import Path
from typing import Optional
import json

from utils.llm_client import LLMClient, LangChainWrapper
from utils.pdf_processor import PDFProcessor
from utils.handlers import (
    extract_section_content,
    extract_section_markers,
    extract_text_between_quotes,
    remove_trailing_dots,
    split_by_subsections,
)
from utils.prompts import (
    prompt_for_second_marker_extraction,
    template_for_subsection_extraction,
    template_for_recomendation_extraction,
    template_for_general_conditions_extraction,
    adapter_system_prompt,
    adapter_user_prompt,
)

app = typer.Typer(help="PDF to JSON converter with optional export functionality")


@app.command()
def convert(
    pdf_path: Path = typer.Argument(..., help="Path to the PDF file", exists=True),
    export: bool = typer.Option(False, "--export", "-e", help="Export JSON to file"),
    output_path: Optional[Path] = typer.Option(
        None,
        "--output",
        "-o",
        help="Output path for JSON file (default: same as PDF with .json extension)",
    ),
):
    """
    Convert a PDF file to JSON structure and optionally export it.
    """

    pdf_name = pdf_path.stem

    try:
        typer.echo(f"Processing PDF: {pdf_path}")

        pdf_processor = PDFProcessor(pdf_path)
        llm_client = LangChainWrapper()
        adapter_llm_client = LLMClient()

        splits = pdf_processor.split_documents()

        try:
            llm_client.create_rag_chain(splits)
            results = llm_client.query_rag_chain()
        except Exception as e:
            typer.echo(f"Error in RAG processing: {str(e)}")
            return None

        if "No treatment heading" in results["answer"]:
            typer.echo("No treatment heading found")
            return None

        start_marker = extract_text_between_quotes(results["answer"])
        results = llm_client.query_rag_chain(
            prompt_for_second_marker_extraction.format(start_marker=start_marker)
        )

        start_marker, end_marker = extract_section_markers(results["answer"])
        typer.echo(f"Start marker: {start_marker}")
        typer.echo(f"End marker: {end_marker}")

        if start_marker is None or end_marker is None:
            typer.echo("No start or end marker found")
            return None

        full_text = pdf_processor.get_full_text()

        extracted_section = extract_section_content(
            full_text, start_marker, remove_trailing_dots(end_marker)
        )

        splits = pdf_processor.split_documents(
            separators=None,
            chunk_size=4000,
            chunk_overlap=0,
            are_docs=False,
            text=extracted_section,
        )

        possible_subsections = []
        last_found_subsection = None
        for split in splits:

            previous_context = (
                last_found_subsection
                if last_found_subsection is not None
                else "No previous subsection found"
            )

            answer_from_llm = llm_client.query_llm(
                template_for_subsection_extraction.format(
                    text=split, previous_context=previous_context
                )
            )

            if "not found" not in answer_from_llm.content.lower():
                possible_subsections.append(answer_from_llm.content)
                last_found_subsection = answer_from_llm.content

        subsections = [start_marker]

        for subsection in possible_subsections:
            if "not found" not in subsection.lower():
                subsections.extend(
                    [splited_subsection.strip() for splited_subsection in subsection.split("\n")]
                )

        subsections_dict = split_by_subsections(extracted_section, subsections)

        typer.echo(f"\nTotal sections found: {len(subsections_dict)}")

        recommendations = []

        for subsection, text in subsections_dict.items():
            general_condition = llm_client.query_llm(
                template_for_general_conditions_extraction.format(text=text)
            )
            answer_from_llm = llm_client.query_llm(
                template_for_recomendation_extraction.format(
                    section_header=subsection, section_text=text
                )
            )
            result_dict = {
                "subsection": subsection,
                "subsection_text": text,
                "general_condition": (
                    general_condition.content
                    if "not found" not in general_condition.content.lower()
                    else None
                ),
                "recommendations": answer_from_llm.content.split("\n\n"),
            }
            recommendations.append(result_dict)

        for recommendation in recommendations:
            general_condition = recommendation["general_condition"]
            if general_condition is not None:
                result_general_condition = adapter_llm_client.query_llm(
                    adapter_system_prompt,
                    adapter_user_prompt.format(text=general_condition),
                )
                recommendation["general_condition"] = result_general_condition

            if "recommendations" in recommendation:
                for recommendation_index in range(len(recommendation["recommendations"])):
                    result_recommendation = adapter_llm_client.query_llm(
                        adapter_system_prompt,
                        adapter_user_prompt.format(
                            text=recommendation["recommendations"][recommendation_index]
                        ),
                    )

                    recommendation_dict = {
                        "recommendation": recommendation["recommendations"][recommendation_index],
                        "condition": json.loads(result_recommendation),
                    }
                    recommendation["recommendations"][recommendation_index] = recommendation_dict

        if export:
            if output_path is None:
                output_path = pdf_path.with_suffix(".json")

            output_data = {
                "pdf_name": pdf_name,
                "nodes": recommendations,
            }

            try:
                with open(output_path, "w") as f:
                    json.dump(output_data, f, indent=4, ensure_ascii=False)
                typer.echo(f"\nData successfully exported to {output_path}")
            except Exception as e:
                typer.echo(f"Error exporting data to JSON: {e}")

    except Exception as e:
        typer.echo(f"Error processing PDF: {str(e)}")
        return None
    return None


if __name__ == "__main__":
    app()
