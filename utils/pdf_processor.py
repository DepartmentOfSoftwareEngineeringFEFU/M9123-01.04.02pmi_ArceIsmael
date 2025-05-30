from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


class PDFProcessor:
    def __init__(self, pdf_path: str):
        try:
            self.loader = PyPDFLoader(pdf_path)
            self.documents = self.loader.load()
        except Exception as e:
            print(f"Error loading PDF: {str(e)}")
            raise

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
        pages = []
        for page in self.documents:
            pages.append(page)
        full_text = " ".join([p.page_content for p in pages])
        return full_text
