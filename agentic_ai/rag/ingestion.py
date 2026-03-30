"""
Document ingestion module for the RAG pipeline.

Handles parsing and extracting content from various document formats
including PDF, plain text, HTML, and Markdown. Provides advanced PDF
parsing capabilities:

    - **Text extraction**: Extracts text content from PDF pages using
      PyMuPDF (fitz) for high-fidelity text extraction.
    - **Image extraction**: Extracts embedded images from PDFs and
      generates captions using multimodal LLM capabilities.
    - **Table parsing**: Detects and extracts tables from PDFs,
      converting them to Markdown format using multimodal LLM
      vision capabilities for accurate structure preservation.

Each parser implements the BaseDocumentParser interface and returns
a list of Document objects ready for the chunking stage.

Requirements:
    pip install pymupdf Pillow

Example:
    >>> from agentic_ai.rag.ingestion import PDFParser, TextFileParser
    >>> parser = PDFParser(llm=my_llm)
    >>> docs = parser.parse("research_paper.pdf")
    >>> for doc in docs:
    ...     print(doc.content[:100])
"""

import base64
import io
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from agentic_ai.core.base_llm import BaseLLM
from agentic_ai.core.models import Document, Message, Role

# Configure module-level logger.
logger = logging.getLogger(__name__)


class BaseDocumentParser(ABC):
    """
    Abstract base class for document parsers.

    All document parsers must implement the ``parse`` method that takes
    a file path and returns a list of Document objects. Each parser
    handles a specific file format.

    Attributes:
        supported_extensions: A set of file extensions this parser supports.
    """

    supported_extensions: set[str] = set()

    @abstractmethod
    def parse(self, file_path: str, **kwargs: Any) -> list[Document]:
        """
        Parse a document file and extract its content.

        Args:
            file_path: Path to the document file to parse.
            **kwargs:  Parser-specific options.

        Returns:
            A list of Document objects extracted from the file.
        """
        ...

    def can_parse(self, file_path: str) -> bool:
        """
        Check if this parser can handle the given file.

        Args:
            file_path: Path to the file to check.

        Returns:
            True if the parser supports this file type.
        """
        suffix = Path(file_path).suffix.lower()
        return suffix in self.supported_extensions


class TextFileParser(BaseDocumentParser):
    """
    Parser for plain text files (.txt, .md, .csv, .log).

    Simply reads the file content and wraps it in a Document object.
    Supports UTF-8 encoding with fallback to latin-1.

    Example:
        >>> parser = TextFileParser()
        >>> docs = parser.parse("notes.txt")
        >>> print(docs[0].content[:50])
    """

    supported_extensions: set[str] = {".txt", ".md", ".csv", ".log", ".rst"}

    def parse(self, file_path: str, **kwargs: Any) -> list[Document]:
        """
        Parse a plain text file into a single Document.

        Args:
            file_path: Path to the text file.
            **kwargs:  Optional ``encoding`` parameter (default: "utf-8").

        Returns:
            A list containing one Document with the file's text content.
        """
        encoding = kwargs.get("encoding", "utf-8")
        path = Path(file_path)

        try:
            content = path.read_text(encoding=encoding)
        except UnicodeDecodeError:
            # Fallback to latin-1 if UTF-8 fails.
            logger.warning(
                "UTF-8 decoding failed for %s, falling back to latin-1", file_path
            )
            content = path.read_text(encoding="latin-1")

        return [
            Document(
                content=content,
                metadata={
                    "source": str(path),
                    "file_name": path.name,
                    "file_type": path.suffix,
                    "parser": "TextFileParser",
                },
            )
        ]


class HTMLParser(BaseDocumentParser):
    """
    Parser for HTML files that extracts readable text content.

    Uses BeautifulSoup to parse HTML and extract text, removing
    scripts, styles, and navigation elements.

    Requirements:
        pip install beautifulsoup4

    Example:
        >>> parser = HTMLParser()
        >>> docs = parser.parse("page.html")
    """

    supported_extensions: set[str] = {".html", ".htm"}

    def parse(self, file_path: str, **kwargs: Any) -> list[Document]:
        """
        Parse an HTML file and extract readable text.

        Args:
            file_path: Path to the HTML file.
            **kwargs:  Unused.

        Returns:
            A list containing one Document with extracted text.
        """
        try:
            from bs4 import BeautifulSoup
        except ImportError as e:
            raise ImportError(
                "The 'beautifulsoup4' package is required for HTMLParser. "
                "Install it with: pip install beautifulsoup4"
            ) from e

        path = Path(file_path)
        html_content = path.read_text(encoding="utf-8")

        soup = BeautifulSoup(html_content, "html.parser")

        # Remove non-content elements.
        for tag in soup(["script", "style", "nav", "header", "footer", "aside"]):
            tag.decompose()

        # Extract text with newlines between block elements.
        text = soup.get_text(separator="\n", strip=True)

        return [
            Document(
                content=text,
                metadata={
                    "source": str(path),
                    "file_name": path.name,
                    "file_type": ".html",
                    "parser": "HTMLParser",
                    "title": soup.title.string if soup.title else "",
                },
            )
        ]


class PDFParser(BaseDocumentParser):
    """
    Advanced PDF parser with text, image, and table extraction.

    Extracts content from PDF files using multiple strategies:

    1. **Text extraction**: Uses PyMuPDF (fitz) for high-fidelity text
       extraction from each page, preserving paragraph structure.

    2. **Image extraction**: Extracts embedded images from PDF pages
       and generates descriptive captions using a multimodal LLM.
       Images are sent as base64-encoded data to the LLM for analysis.

    3. **Table extraction**: Detects table-like structures on pages
       and uses multimodal LLM vision to convert them into clean
       Markdown table format, preserving rows and columns.

    Attributes:
        llm: Optional multimodal LLM for image captioning and table parsing.
        extract_images: Whether to extract and caption images.
        extract_tables: Whether to extract and parse tables.
        min_image_size: Minimum image dimension (px) to process.

    Requirements:
        pip install pymupdf Pillow

    Example:
        >>> from agentic_ai.llms.openai_llm import OpenAIChatModel
        >>> llm = OpenAIChatModel(api_key="sk-...", model="gpt-4o")
        >>> parser = PDFParser(llm=llm, extract_images=True, extract_tables=True)
        >>> docs = parser.parse("research_paper.pdf")
    """

    supported_extensions: set[str] = {".pdf"}

    def __init__(
        self,
        llm: BaseLLM | None = None,
        extract_images: bool = True,
        extract_tables: bool = True,
        min_image_size: int = 100,
    ) -> None:
        """
        Initialize the PDF parser.

        Args:
            llm:             A multimodal LLM for image captioning and table parsing.
                             Required if extract_images or extract_tables is True.
            extract_images:  Whether to extract images and generate captions.
                             Default is True.
            extract_tables:  Whether to extract tables as Markdown.
                             Default is True.
            min_image_size:  Minimum width or height (in pixels) for an image
                             to be processed. Smaller images are skipped.
                             Default is 100.
        """
        self.llm = llm
        self.extract_images = extract_images
        self.extract_tables = extract_tables
        self.min_image_size = min_image_size

        if (extract_images or extract_tables) and llm is None:
            logger.warning(
                "PDFParser: extract_images/extract_tables is enabled but no LLM "
                "provided. Image captioning and table parsing will be skipped."
            )

    def parse(self, file_path: str, **kwargs: Any) -> list[Document]:
        """
        Parse a PDF file and extract text, images, and tables.

        Processes each page of the PDF sequentially. For each page:
        1. Extracts text content.
        2. If enabled, extracts images and generates captions via LLM.
        3. If enabled, renders the page and detects tables via LLM.

        Args:
            file_path: Path to the PDF file.
            **kwargs:  Optional parameters:
                       - ``pages``: List of page numbers to parse (0-indexed).
                         If not provided, all pages are parsed.

        Returns:
            A list of Document objects, one per page, with text content
            and any extracted image captions or table Markdown appended.
        """
        try:
            import fitz  # PyMuPDF
        except ImportError as e:
            raise ImportError(
                "The 'pymupdf' package is required for PDFParser. "
                "Install it with: pip install pymupdf"
            ) from e

        path = Path(file_path)
        doc = fitz.open(str(path))
        documents: list[Document] = []

        # Determine which pages to process.
        pages_to_parse: list[int] = kwargs.get("pages", list(range(len(doc))))

        logger.info("Parsing PDF: %s (%d pages)", file_path, len(doc))

        for page_num in pages_to_parse:
            if page_num >= len(doc):
                logger.warning("Page %d out of range, skipping", page_num)
                continue

            page = doc[page_num]
            content_parts: list[str] = []

            # --- 1. Text Extraction ---
            page_text = page.get_text("text")
            if page_text.strip():
                content_parts.append(page_text.strip())

            # --- 2. Image Extraction and Captioning ---
            if self.extract_images and self.llm:
                image_captions = self._extract_and_caption_images(page, page_num)
                if image_captions:
                    content_parts.append("\n--- Extracted Images ---")
                    content_parts.extend(image_captions)

            # --- 3. Table Extraction ---
            if self.extract_tables and self.llm:
                tables_md = self._extract_tables_from_page(page, page_num, fitz)
                if tables_md:
                    content_parts.append("\n--- Extracted Tables ---")
                    content_parts.append(tables_md)

            # Combine all content parts for this page.
            full_content = "\n\n".join(content_parts)

            if full_content.strip():
                documents.append(
                    Document(
                        content=full_content,
                        metadata={
                            "source": str(path),
                            "file_name": path.name,
                            "file_type": ".pdf",
                            "page_number": page_num,
                            "total_pages": len(doc),
                            "parser": "PDFParser",
                        },
                    )
                )

        doc.close()
        logger.info("Extracted %d documents from PDF: %s", len(documents), file_path)
        return documents

    def _extract_and_caption_images(
        self, page: Any, page_num: int
    ) -> list[str]:
        """
        Extract images from a PDF page and generate captions using LLM.

        Iterates through all images embedded in the page, filters by
        minimum size, converts to base64, and sends to the multimodal
        LLM for captioning.

        Args:
            page:     The PyMuPDF page object.
            page_num: The page number (for logging).

        Returns:
            A list of caption strings for extracted images.
        """
        try:
            from PIL import Image as PILImage
        except ImportError:
            logger.warning("Pillow not installed, skipping image extraction")
            return []

        captions: list[str] = []
        image_list = page.get_images(full=True)

        for img_idx, img_info in enumerate(image_list):
            xref = img_info[0]

            try:
                # Extract the image data from the PDF.
                base_image = page.parent.extract_image(xref)
                image_bytes = base_image["image"]

                # Open with PIL to check dimensions.
                pil_image = PILImage.open(io.BytesIO(image_bytes))
                width, height = pil_image.size

                # Skip images smaller than the minimum size.
                if width < self.min_image_size or height < self.min_image_size:
                    continue

                # Convert image to base64 for the LLM.
                buffered = io.BytesIO()
                pil_image.save(buffered, format="PNG")
                img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

                # Generate a caption using the multimodal LLM.
                caption = self._caption_image(img_base64, page_num, img_idx)
                if caption:
                    captions.append(f"[Image {img_idx + 1}]: {caption}")

            except Exception as e:
                logger.warning(
                    "Failed to extract image %d from page %d: %s",
                    img_idx, page_num, e,
                )

        return captions

    def _caption_image(self, img_base64: str, page_num: int, img_idx: int) -> str:
        """
        Generate a descriptive caption for an image using a multimodal LLM.

        Sends the base64-encoded image to the LLM with a prompt asking
        for a detailed description of the image content.

        Args:
            img_base64: Base64-encoded PNG image data.
            page_num:   Page number the image was found on.
            img_idx:    Index of the image on the page.

        Returns:
            A descriptive caption string, or empty string on failure.
        """
        if not self.llm:
            return ""

        try:
            # Build a multimodal message with the image.
            messages = [
                Message(
                    role=Role.USER,
                    content=(
                        f"Describe this image from page {page_num + 1} of a PDF document. "
                        "Provide a detailed but concise description of what the image shows, "
                        "including any text, diagrams, charts, or visual elements. "
                        "Focus on information that would be useful for understanding the "
                        "document's content."
                    ),
                    metadata={"image_base64": img_base64},
                )
            ]
            response = self.llm.chat(messages)
            return response.content

        except Exception as e:
            logger.warning(
                "Failed to caption image %d on page %d: %s",
                img_idx, page_num, e,
            )
            return ""

    def _extract_tables_from_page(
        self, page: Any, page_num: int, fitz_module: Any
    ) -> str:
        """
        Extract tables from a PDF page using multimodal LLM vision.

        Renders the entire page as an image and sends it to the LLM
        with a prompt to detect and extract any tables in Markdown format.

        Args:
            page:        The PyMuPDF page object.
            page_num:    The page number (for logging).
            fitz_module: The fitz (PyMuPDF) module reference.

        Returns:
            Markdown-formatted table string, or empty string if no tables found.
        """
        if not self.llm:
            return ""

        try:
            # Render the page as a high-resolution image.
            # Use a 2x zoom for better table detection.
            mat = fitz_module.Matrix(2, 2)
            pix = page.get_pixmap(matrix=mat)
            img_bytes = pix.tobytes("png")

            # Convert to base64.
            img_base64 = base64.b64encode(img_bytes).decode("utf-8")

            # Ask the LLM to extract tables.
            messages = [
                Message(
                    role=Role.USER,
                    content=(
                        f"Look at this image of page {page_num + 1} from a PDF document. "
                        "If there are any tables in this image, extract them and format "
                        "them as Markdown tables. Preserve the exact structure with all "
                        "rows and columns. If there are no tables, respond with 'NO_TABLES'. "
                        "Only output the Markdown tables, nothing else."
                    ),
                    metadata={"image_base64": img_base64},
                )
            ]
            response = self.llm.chat(messages)

            # Check if the LLM found any tables.
            if "NO_TABLES" in response.content.upper():
                return ""

            return response.content

        except Exception as e:
            logger.warning(
                "Failed to extract tables from page %d: %s", page_num, e
            )
            return ""


class DocumentIngestionPipeline:
    """
    Orchestrates document ingestion from multiple file types.

    Automatically selects the appropriate parser based on file extension
    and processes documents through the ingestion pipeline. Supports
    batch processing of multiple files.

    Attributes:
        parsers: A list of registered document parsers.

    Example:
        >>> pipeline = DocumentIngestionPipeline(llm=my_llm)
        >>> docs = pipeline.ingest(["paper.pdf", "notes.txt", "page.html"])
    """

    def __init__(
        self,
        llm: BaseLLM | None = None,
        extract_images: bool = True,
        extract_tables: bool = True,
    ) -> None:
        """
        Initialize the ingestion pipeline with default parsers.

        Args:
            llm:            Optional multimodal LLM for PDF image/table extraction.
            extract_images: Whether to extract images from PDFs.
            extract_tables: Whether to extract tables from PDFs.
        """
        # Register default parsers.
        self.parsers: list[BaseDocumentParser] = [
            TextFileParser(),
            HTMLParser(),
            PDFParser(
                llm=llm,
                extract_images=extract_images,
                extract_tables=extract_tables,
            ),
        ]

    def add_parser(self, parser: BaseDocumentParser) -> None:
        """
        Register an additional document parser.

        Args:
            parser: A BaseDocumentParser implementation to add.
        """
        self.parsers.append(parser)

    def ingest(self, file_paths: list[str], **kwargs: Any) -> list[Document]:
        """
        Ingest documents from multiple file paths.

        Automatically selects the appropriate parser for each file based
        on its extension. Files with unsupported extensions are skipped
        with a warning.

        Args:
            file_paths: A list of file paths to ingest.
            **kwargs:   Additional arguments passed to each parser.

        Returns:
            A flat list of Document objects from all parsed files.
        """
        all_documents: list[Document] = []

        for file_path in file_paths:
            # Find a parser that supports this file type.
            parser = self._find_parser(file_path)

            if parser is None:
                logger.warning("No parser found for file: %s", file_path)
                continue

            try:
                docs = parser.parse(file_path, **kwargs)
                all_documents.extend(docs)
                logger.info(
                    "Ingested %d documents from: %s", len(docs), file_path
                )
            except Exception as e:
                logger.error("Failed to ingest %s: %s", file_path, e)

        logger.info(
            "Ingestion complete: %d documents from %d files",
            len(all_documents), len(file_paths),
        )
        return all_documents

    def _find_parser(self, file_path: str) -> BaseDocumentParser | None:
        """
        Find a parser that can handle the given file.

        Args:
            file_path: Path to the file.

        Returns:
            A matching parser, or None if no parser supports this file type.
        """
        for parser in self.parsers:
            if parser.can_parse(file_path):
                return parser
        return None
