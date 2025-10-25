"""Document processor for parsing and chunking markdown files."""

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from langchain_text_splitters import RecursiveCharacterTextSplitter


@dataclass
class Chunk:
    """Represents a chunk of text with metadata."""

    content: str
    metadata: Dict[str, str]
    chunk_id: str
    parent_id: Optional[str] = None


@dataclass
class Document:
    """Represents a document with its chunks."""

    document_name: str
    file_path: str
    parent_chunks: List[Chunk]
    child_chunks: List[Chunk]
    metadata: Dict[str, str]


class MarkdownParser:
    """Parser for markdown files with header extraction."""

    def __init__(self):
        """Initialize the markdown parser."""
        self.header_pattern = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)

    def parse_markdown(self, content: str, file_path: str) -> Dict[str, any]:
        """
        Parse markdown content and extract headers with their content.

        Args:
            content: The markdown content to parse.
            file_path: Path to the source file.

        Returns:
            Dict containing parsed sections with headers and content.
        """
        lines = content.split("\n")
        sections = []
        current_section = {"headers": [], "content": [], "level": 0}

        for line in lines:
            header_match = self.header_pattern.match(line)
            if header_match:
                # Save previous section if it has content
                if current_section["content"]:
                    sections.append(current_section.copy())

                # Start new section
                level = len(header_match.group(1))
                header_text = header_match.group(2).strip()

                # Update headers based on level
                if level == 1:
                    current_section["headers"] = [header_text]
                elif level > len(current_section["headers"]):
                    current_section["headers"].append(header_text)
                else:
                    current_section["headers"] = (
                        current_section["headers"][: level - 1] + [header_text]
                    )

                current_section["level"] = level
                current_section["content"] = [line]
            else:
                current_section["content"].append(line)

        # Add the last section
        if current_section["content"]:
            sections.append(current_section)

        return {
            "file_path": file_path,
            "sections": sections,
            "full_content": content,
        }


class DocumentProcessor:
    """Process documents into parent and child chunks."""

    def __init__(
        self,
        parent_chunk_size: int = 2000,
        parent_chunk_overlap: int = 200,
        child_chunk_size: int = 500,
        child_chunk_overlap: int = 50,
    ):
        """
        Initialize document processor.

        Args:
            parent_chunk_size: Size of parent chunks (sections).
            parent_chunk_overlap: Overlap between parent chunks.
            child_chunk_size: Size of child chunks (paragraphs).
            child_chunk_overlap: Overlap between child chunks.
        """
        self.parent_chunk_size = parent_chunk_size
        self.parent_chunk_overlap = parent_chunk_overlap
        self.child_chunk_size = child_chunk_size
        self.child_chunk_overlap = child_chunk_overlap

        self.parser = MarkdownParser()

        self.parent_splitter = RecursiveCharacterTextSplitter(
            chunk_size=parent_chunk_size,
            chunk_overlap=parent_chunk_overlap,
            separators=["\n## ", "\n### ", "\n#### ", "\n\n", "\n", " ", ""],
        )

        self.child_splitter = RecursiveCharacterTextSplitter(
            chunk_size=child_chunk_size,
            chunk_overlap=child_chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

    def process_document(self, file_path: str) -> Document:
        """
        Process a markdown document into parent and child chunks.

        Args:
            file_path: Path to the markdown file.

        Returns:
            Document object containing all chunks with metadata.
        """
        path = Path(file_path)
        document_name = path.stem

        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        # Parse markdown to extract headers
        parsed = self.parser.parse_markdown(content, file_path)

        # Create parent chunks
        parent_chunks = []
        child_chunks = []

        for section_idx, section in enumerate(parsed["sections"]):
            section_content = "\n".join(section["content"])
            headers = section["headers"]

            # Create metadata from headers
            metadata = {
                "document_name": document_name,
                "file_path": file_path,
                "section_idx": str(section_idx),
            }

            # Add header hierarchy to metadata
            for i, header in enumerate(headers):
                metadata[f"header_level_{i+1}"] = header

            # Create parent chunk for this section
            parent_chunk_id = f"{document_name}_parent_{section_idx}"
            parent_chunk = Chunk(
                content=section_content,
                metadata=metadata.copy(),
                chunk_id=parent_chunk_id,
                parent_id=None,
            )
            parent_chunks.append(parent_chunk)

            # Split section into child chunks
            child_texts = self.child_splitter.split_text(section_content)

            for child_idx, child_text in enumerate(child_texts):
                child_chunk_id = f"{document_name}_child_{section_idx}_{child_idx}"
                child_metadata = metadata.copy()
                child_metadata["child_idx"] = str(child_idx)

                child_chunk = Chunk(
                    content=child_text,
                    metadata=child_metadata,
                    chunk_id=child_chunk_id,
                    parent_id=parent_chunk_id,
                )
                child_chunks.append(child_chunk)

        return Document(
            document_name=document_name,
            file_path=file_path,
            parent_chunks=parent_chunks,
            child_chunks=child_chunks,
            metadata={"document_name": document_name, "file_path": file_path},
        )

    def process_documents(self, file_paths: List[str]) -> List[Document]:
        """
        Process multiple markdown documents.

        Args:
            file_paths: List of paths to markdown files.

        Returns:
            List of Document objects.
        """
        documents = []
        for file_path in file_paths:
            try:
                doc = self.process_document(file_path)
                documents.append(doc)
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
                continue
        return documents
