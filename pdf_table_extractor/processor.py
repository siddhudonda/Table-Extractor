from pathlib import Path
from typing import Dict, List, Optional, Any
from tqdm import tqdm
import logging
from .core import EnhancedPDFExtractor
from .utils import validate_pdf_path

logger = logging.getLogger("PDFTableExtractor")

class PDFProcessor:
    """Batch process multiple PDFs.

    Args:
        extractor: EnhancedPDFExtractor instance.
        output_base_dir: Base directory for output files.
    """
    def __init__(self, extractor: EnhancedPDFExtractor, output_base_dir: str = "processed_pdfs"):
        self.extractor = extractor
        self.output_base_dir = output_base_dir
    
    def process_directory(self, input_dir: str, page_numbers: Optional[List[int]] = None,
                         strategy: Optional[str] = None, output_format: str = "csv",
                         password: Optional[str] = None) -> Dict[str, Dict[str, Any]]:
        """Process all PDFs in a directory.

        Args:
            input_dir: Directory containing PDFs.
            page_numbers: List of page numbers (0-based) to process.
            strategy: Table extraction strategy.
            output_format: Format for saved tables ('csv' or 'json').
            password: Password for encrypted PDFs.

        Returns:
            Dictionary mapping filenames to results.
        """
        results = {}
        Path(self.output_base_dir).mkdir(parents=True, exist_ok=True)
        pdf_files = [f for f in Path(input_dir).glob("*.pdf")]
        
        for pdf_path in tqdm(pdf_files, desc="Processing PDFs"):
            output_dir = Path(self.output_base_dir) / pdf_path.stem
            try:
                results[pdf_path.name] = self.extractor.process(
                    str(pdf_path), str(output_dir), page_numbers, strategy, output_format, password
                )
            except Exception as e:
                logger.error(f"Failed to process {pdf_path.name}: {e}")
                results[pdf_path.name] = {"error": str(e)}
        return results
    
    def summarize_results(self, results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Summarize processing results.

        Args:
            results: Dictionary of processing results.

        Returns:
            Summary statistics.
        """
        summary = {"total_files": len(results), "successful": 0, "failed": 0, "total_tables": 0, "total_pages": 0}
        for filename, result in results.items():
            if "error" in result:
                summary["failed"] += 1
                continue
            summary["successful"] += 1
            if "tables" in result:
                for page_num, tables in result["tables"].items():
                    summary["total_tables"] += len(tables)
                    summary["total_pages"] += 1
        return summary