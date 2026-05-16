import logging
from pathlib import Path
from typing import List

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader


from src.config import settings

logger = logging.getLogger(__name__)
def load_documents(data_dir: Path | None = None) -> List[Document]:
    """Load all PDFs and text files from the data directory."""
    data_dir = data_dir or settings.data_dir

    docs = []

    # --- Load PDFs ---
    pdf_files = list(data_dir.rglob("*.pdf"))

    if pdf_files:
        for pdf_path in pdf_files:
            try:
                loader = PyPDFLoader(str(pdf_path))
                pages = loader.load()

                for page in pages:
                    page.metadata["source"] = str(pdf_path)
                    page.metadata["company"] = _infer_company(pdf_path.name)

                docs.extend(pages)
                logger.info("Loaded %d pages from %s", len(pages), pdf_path.name)

            except Exception as e:
                logger.warning("Could not load %s: %s", pdf_path, e)

    # --- Fallback to sample data if no files found ---
    if not docs:
        logger.warning("No documents found — using built-in sample data")
        docs = _sample_documents()

    logger.info("Total documents loaded: %d", len(docs))
    return docs


def chunk_documents(docs: List[Document]) -> List[Document]:
    """Split documents into overlapping chunks."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks = splitter.split_documents(docs)
    logger.info("Split into %d chunks", len(chunks))
    return chunks


def _infer_company(filename: str) -> str:
    """Guess the company from the filename."""
    name = filename.lower()
    if "alliander" in name:
        return "Alliander"
    if "enexis" in name:
        return "Enexis"
    return "Unknown"


def _sample_documents() -> List[Document]:
    """Built-in sample data so the app works without real PDFs."""
    passages = [
        {
            "content": (
                "Alliander is the largest energy network operator in the Netherlands, "
                "serving approximately 3.4 million customers. In 2023, Alliander reported "
                "total revenues of €2.1 billion and a net profit of €312 million."
            ),
            "source": "alliander_sample.txt",
            "company": "Alliander",
            "page": 1,
        },
        {
            "content": (
                "Alliander aims to reduce its CO₂ emissions by 49% by 2030 compared "
                "to 2015 levels, and achieve full carbon neutrality by 2050. "
                "The company invested €1.4 billion in grid infrastructure in 2023."
            ),
            "source": "alliander_sample.txt",
            "company": "Alliander",
            "page": 12,
        },
        {
            "content": (
                "Enexis Holding N.V. serves approximately 2.9 million electricity "
                "customers across the northern, eastern and southern Netherlands. "
                "In 2023, Enexis reported revenues of €1.8 billion."
            ),
            "source": "enexis_sample.txt",
            "company": "Enexis",
            "page": 1,
        },
        {
            "content": (
                "Enexis invested €1.1 billion in grid infrastructure in 2023, a 22% "
                "increase from 2022. The company aims to reduce its greenhouse gas "
                "emissions by 55% by 2030 compared to 2019 levels."
            ),
            "source": "enexis_sample.txt",
            "company": "Enexis",
            "page": 18,
        },
        {
            "content": (
                "Grid congestion is the most pressing challenge for Dutch energy network "
                "operators in 2023. Both Alliander and Enexis report multi-year waiting "
                "lists for new grid connections, particularly for solar farms and "
                "industrial customers."
            ),
            "source": "sector_context.txt",
            "company": "Sector",
            "page": 1,
        },
    ]

    return [
        Document(
            page_content=p["content"],
            metadata={
                "source": p["source"],
                "company": p["company"],
                "page": p["page"],
            },
        )
        for p in passages
    ]

