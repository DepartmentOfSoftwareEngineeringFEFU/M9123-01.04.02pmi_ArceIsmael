import io
import sys
from contextlib import redirect_stderr
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


class PDFCorruptionError(Exception):
    """Custom exception for corrupted PDF files"""

    pass


class PDFProcessor:
    def __init__(self, pdf_path: str):
        self.pdf_path = pdf_path
        self.corruption_warnings = []

        try:
            stderr_capture = io.StringIO()

            with redirect_stderr(stderr_capture):
                self.loader = PyPDFLoader(pdf_path)
                self.documents = self.loader.load()

            stderr_output = stderr_capture.getvalue()
            if stderr_output:
                corruption_indicators = [
                    "Ignoring wrong pointing object",
                    "Invalid PDF structure",
                    "Error reading PDF",
                    "Corrupted PDF",
                    "Invalid xref entry",
                ]

                for indicator in corruption_indicators:
                    if indicator.lower() in stderr_output.lower():
                        self.corruption_warnings.append(stderr_output.strip())
                        break

            self._validate_pdf_content()

        except Exception as e:
            error_msg = str(e).lower()
            if any(
                keyword in error_msg for keyword in ["corrupt", "invalid", "malformed", "damaged"]
            ):
                raise PDFCorruptionError(f"PDF file appears to be corrupted: {str(e)}")
            else:
                print(f"Error loading PDF: {str(e)}")
                raise

    def _validate_pdf_content(self):
        """Validate that PDF content was extracted properly"""
        if not self.documents:
            raise PDFCorruptionError(
                "No content could be extracted from PDF - file may be corrupted"
            )

        total_chars = sum(len(doc.page_content) for doc in self.documents)
        if total_chars < 100:
            raise PDFCorruptionError("PDF content appears incomplete - file may be corrupted")

        if len(self.corruption_warnings) > 0:
            corruption_count = sum(
                line.count("Ignoring wrong pointing object") for line in self.corruption_warnings
            )

            if corruption_count > 10:
                raise PDFCorruptionError(
                    f"PDF file has extensive corruption ({corruption_count} corrupted objects detected). "
                    f"Please provide a clean version of the PDF file: {self.pdf_path}"
                )
            elif corruption_count > 0:
                print(
                    f"⚠️  WARNING: PDF file has {corruption_count} corrupted objects but processing will continue"
                )

    def split_documents(
        self,
        separators: list[str] | None = ["\n\n", "\n", " ", ""],
        chunk_size: int = 4000,
        chunk_overlap: int = 200,
        are_docs: bool = True,
        text: str | None = None,
    ):
        try:
            text_splitter = RecursiveCharacterTextSplitter(
                separators=separators,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )
            if are_docs:
                if not self.documents:
                    raise ValueError("No documents loaded from PDF")
                splits = text_splitter.split_documents(self.documents)
                return splits
            else:
                if text is None:
                    raise ValueError("Text parameter is required when are_docs is False")
                return text_splitter.split_text(text)
        except Exception as e:
            print(f"Error in split_documents: {str(e)}")
            raise

    def get_full_text(self):
        if not self.documents:
            raise PDFCorruptionError("No documents available - PDF may be corrupted")

        pages = []
        for page in self.documents:
            pages.append(page)
        full_text = " ".join([p.page_content for p in pages])

        # Additional validation of extracted text
        if len(full_text.strip()) < 100:
            raise PDFCorruptionError("Extracted text is too short - PDF may be corrupted")

        return full_text

    def is_corrupted(self) -> bool:
        """Check if PDF shows signs of corruption"""
        return len(self.corruption_warnings) > 0
