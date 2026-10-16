
from pathlib import Path


SUPPORTED_EXTENSIONS = {".txt", ".pdf"}


def extract_documents(data_dir: str | Path) -> list[dict]:
    """Extract text while preserving source filenames and PDF pages."""
    data_dir = Path(data_dir)

    if not data_dir.exists():
        raise FileNotFoundError(f"Directory not found: {data_dir}")

    documents = []

    for file_path in sorted(data_dir.rglob("*")):
        if not file_path.is_file():
            continue

        if file_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue

        if file_path.suffix.lower() == ".txt":
            text = file_path.read_text(encoding="utf-8").strip()

            if text:
                documents.append({
                    "text": text,
                    "source": file_path.name,
                    "page": None,
                })

        elif file_path.suffix.lower() == ".pdf":
            import fitz  # PyMuPDF

            with fitz.open(file_path) as pdf:
                for page_number, page in enumerate(pdf, start=1):
                    text = page.get_text().strip()

                    if text:
                        documents.append({
                            "text": text,
                            "source": file_path.name,
                            "page": page_number,
                        })

    if not documents:
        raise ValueError(
            f"No readable text documents found in {data_dir}"
        )

    return documents


def chunk_documents(
    documents: list[dict],
    chunk_size: int = 800,
    chunk_overlap: int = 150,
) -> list[dict]:
    """Split documents into overlapping character-based chunks."""

    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero.")

    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError(
            "chunk_overlap must be non-negative and smaller than chunk_size."
        )

    chunks = []
    step = chunk_size - chunk_overlap

    for document in documents:
        text = document["text"].strip()

        for start in range(0, len(text), step):
            chunk_text = text[start:start + chunk_size].strip()

            if not chunk_text:
                continue

            chunks.append({
                "chunk_id": f"chunk_{len(chunks) + 1}",
                "text": chunk_text,
                "source": document["source"],
                "page": document.get("page"),
            })

    return chunks


if __name__ == "__main__":
    data_dir = Path(__file__).resolve().parent / "data"

    documents = extract_documents(data_dir)
    chunks = chunk_documents(documents)

    print(f"Documents extracted: {len(documents)}")
    print(f"Chunks created: {len(chunks)}")

    for chunk in chunks:
        print(
            f"\nID: {chunk['chunk_id']} | "
            f"Source: {chunk['source']} | "
            f"Page: {chunk['page']}"
        )
        print(chunk["text"][:150])
