import logging
import os

def setup_logging(verbosity: str = "INFO") -> None:
    """Configure logging with specified verbosity."""
    levels = {"DEBUG": logging.DEBUG, "INFO": logging.INFO, "WARNING": logging.WARNING, "ERROR": logging.ERROR}
    logging.basicConfig(
        level=levels.get(verbosity.upper(), logging.INFO),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

def validate_pdf_path(file_path: str) -> bool:
    """Validate that a file exists and is a PDF."""
    if not os.path.isfile(file_path) or not file_path.lower().endswith('.pdf'):
        logging.error(f"Invalid PDF path: {file_path}")
        return False
    return True