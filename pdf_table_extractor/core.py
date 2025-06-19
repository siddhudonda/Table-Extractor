import fitz
import json
import csv
from typing import Dict, List, Optional, Any
from multiprocessing import Pool, cpu_count
from functools import partial
from tqdm import tqdm
from pathlib import Path
import logging
from .extractors import Table, TableExtractor, MuPDFTableExtractor, GridBasedTableExtractor, TextBasedTableExtractor

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False

try:
    from rtree import index
    RTREE_AVAILABLE = True
except ImportError:
    RTREE_AVAILABLE = False

logger = logging.getLogger("PDFTableExtractor")

class EnhancedPDFExtractor:
    """Extract tables and text from PDFs with advanced capabilities.

    Args:
        debug (bool): Enable debug visualizations.
        strategy (str): Table extraction strategy ('auto', 'mupdf', 'grid', 'text', 'all').
        continuous (bool): Merge tables across consecutive pages.
    """
    def __init__(self, debug: bool = False, strategy: str = "auto", continuous: bool = True):
        self.debug = debug
        self.strategy = strategy.lower()
        self.continuous = continuous
        self.extractors: Dict[str, TableExtractor] = {
            "mupdf": MuPDFTableExtractor(debug=debug),
            "grid": GridBasedTableExtractor(debug=debug),
            "text": TextBasedTableExtractor(debug=debug)
        }
    
    def choose_best_strategy(self, page: fitz.Page) -> str:
        """Select the best extraction strategy based on page content."""
        lines = page.get_drawings()
        has_grid = len(lines) > 10
        text_blocks = page.get_text("words")
        text_density = len(text_blocks) / (page.rect.width * page.rect.height + 1e-6)
        if has_grid:
            return "grid"
        elif text_density > 0.001:
            return "text"
        return "mupdf"
    
    def open(self, file_path: str, password: Optional[str] = None) -> fitz.Document:
        """Open a PDF file."""
        try:
            return fitz.open(file_path, password=password)
        except Exception as e:
            logger.error(f"Error opening PDF: {e}")
            raise
    
    def extract_text(self, document_or_page: Any, page_numbers: Optional[List[int]] = None) -> Dict[int, Dict[str, Any]]:
        """Extract raw and structured text from a PDF or page.

        Args:
            document_or_page: fitz.Document or fitz.Page object.
            page_numbers: List of page numbers (0-based) to process.

        Returns:
            Dictionary mapping page numbers to text data.
        """
        result = {}
        try:
            if isinstance(document_or_page, fitz.Document):
                pages = range(document_or_page.page_count) if page_numbers is None else page_numbers
                for page_num in pages:
                    if 0 <= page_num < document_or_page.page_count:
                        page = document_or_page[page_num]
                        raw_text = page.get_text()
                        structured = page.get_text("dict")
                        result[page_num] = {"raw": raw_text, "structured": structured}
            elif isinstance(document_or_page, fitz.Page):
                page = document_or_page
                raw_text = page.get_text()
                structured = page.get_text("dict")
                result[page.number] = {"raw": raw_text, "structured": structured}
            else:
                raise TypeError("Expected Document or Page object")
        except Exception as e:
            logger.error(f"Error extracting text: {e}")
        return result
    
    def _process_page(self, page_num: int, doc_path: str, strategy: str) -> tuple[int, List[Table], float]:
        """Process a single page (for multiprocessing)."""
        try:
            doc = fitz.open(doc_path)
            page = doc.load_page(page_num)
            rotation = page.rotation
            strategy = strategy if strategy != "auto" else self.choose_best_strategy(page)
            tables = []
            if strategy == "all":
                for extractor in self.extractors.values():
                    tables.extend(extractor.extract(page, rotation))
                tables = self._eliminate_overlapping_tables(tables)
            else:
                tables = self.extractors[strategy].extract(page, rotation)
            doc.close()
            return page_num, tables, rotation
        except Exception as e:
            logger.error(f"Error processing page {page_num}: {e}")
            return page_num, [], 0
    
    def _merge_continuous_tables(self, tables: Dict[int, List[Table]]) -> Dict[int, List[Table]]:
        """Merge tables that continue across pages."""
        if not self.continuous:
            return tables
        
        merged_tables = []
        page_nums = sorted(tables.keys())
        
        for i, page_num in enumerate(page_nums):
            for table in tables[page_num]:
                table.continuous_pages = [page_num]
                if not merged_tables:
                    merged_tables.append(table)
                    continue
                
                last_table = merged_tables[-1]
                is_continuous = False
                if page_num == last_table.continuous_pages[-1] + 1:
                    if last_table.header_rows > 0 and table.header_rows > 0:
                        last_header = [c.text for c in last_table.rows[0].cells]
                        curr_header = [c.text for c in table.rows[0].cells]
                        if last_header == curr_header:
                            is_continuous = True
                    if not is_continuous:
                        last_bbox = last_table.bbox
                        curr_bbox = table.bbox
                        from math import isclose
                        if (isclose(last_bbox[0], curr_bbox[0], abs_tol=10) and
                            isclose(last_bbox[2], curr_bbox[2], abs_tol=10)):
                            is_continuous = True
                
                if is_continuous:
                    logger.info(f"Merging table from page {page_num} with table ending on page {last_table.continuous_pages[-1]}")
                    start_row = table.header_rows
                    for row in table.rows[start_row:]:
                        row.cells = [TableCell(
                            text=c.text, bbox=c.bbox, row_idx=len(last_table.rows) + c.row_idx - start_row,
                            col_idx=c.col_idx, is_header=False
                        ) for c in row.cells]
                        last_table.rows.append(row)
                    last_table.continuous_pages.append(page_num)
                    last_table.bbox = (
                        min(last_table.bbox[0], table.bbox[0]),
                        min(last_table.bbox[1], table.bbox[1]),
                        max(last_table.bbox[2], table.bbox[2]),
                        max(last_table.bbox[3], table.bbox[3])
                    )
                else:
                    merged_tables.append(table)
        
        result = {}
        for table in merged_tables:
            first_page = table.continuous_pages[0]
            if first_page not in result:
                result[first_page] = []
            result[first_page].append(table)
        
        return result
    
    def extract_tables(self, document_or_page: Any, page_numbers: Optional[List[int]] = None, 
                      strategy: Optional[str] = None) -> Dict[int, List[Table]]:
        """Extract tables from a PDF or page.

        Args:
            document_or_page: fitz.Document or fitz.Page object.
            page_numbers: List of page numbers (0-based) to process.
            strategy: Extraction strategy to override default.

        Returns:
            Dictionary mapping page numbers to lists of tables.
        """
        result = {}
        strategy = strategy or self.strategy
        try:
            if isinstance(document_or_page, fitz.Document):
                doc = document_or_page
                doc_path = doc.name if hasattr(doc, 'name') else str(Path.cwd() / "temp.pdf")
                pages = range(doc.page_count) if page_numbers is None else [p for p in page_numbers if 0 <= p < doc.page_count]
                
                with Pool(processes=cpu_count()) as pool:
                    process_func = partial(self._process_page, doc_path=doc_path, strategy=strategy)
                    results = list(tqdm(pool.imap(process_func, pages), total=len(pages), desc="Processing pages"))
                
                for page_num, tables, _ in results:
                    if tables:
                        result[page_num] = tables
                result = self._merge_continuous_tables(result)
            elif isinstance(document_or_page, fitz.Page):
                page = document_or_page
                rotation = page.rotation
                strategy = strategy if strategy != "auto" else self.choose_best_strategy(page)
                if strategy == "all":
                    tables = []
                    for extractor in self.extractors.values():
                        tables.extend(extractor.extract(page, rotation))
                    result[page.number] = self._eliminate_overlapping_tables(tables)
                else:
                    result[page.number] = self.extractors[strategy].extract(page, rotation)
            else:
                raise TypeError("Expected Document or Page object")
        except Exception as e:
            logger.error(f"Error extracting tables: {e}")
        return result
    
    def _eliminate_overlapping_tables(self, tables: List[Table]) -> List[Table]:
        """Remove overlapping tables, keeping the best ones."""
        if not tables:
            return []
        if RTREE_AVAILABLE:
            idx = index.Index()
            keep = []
            tables = sorted(tables, key=lambda t: t.num_rows * t.num_cols, reverse=True)
            for i, table in enumerate(tables):
                x0, y0, x1, y1 = table.bbox
                if not list(idx.intersection((x0, y0, x1, y1))):
                    idx.insert(i, (x0, y0, x1, y1))
                    keep.append(table)
            return keep
        result = []
        used_areas = []
        tables = sorted(tables, key=lambda t: t.num_rows * t.num_cols, reverse=True)
        for table in tables:
            overlap = False
            for used_bbox in used_areas:
                x0_1, y0_1, x1_1, y1_1 = table.bbox
                x0_2, y0_2, x1_2, y1_2 = used_bbox
                if not (x1_1 <= x0_2 or x1_2 <= x0_1 or y1_1 <= y0_2 or y1_2 <= y0_1):
                    intersection = (min(x1_1, x1_2) - max(x0_1, x0_2)) * (min(y1_1, y1_2) - max(y0_1, y0_2))
                    area1 = (x1_1 - x0_1) * (y1_1 - y0_1)
                    if intersection > 0.5 * area1:
                        overlap = True
                        break
            if not overlap:
                result.append(table)
                used_areas.append(table.bbox)
        return result
    
    def save_tables(self, tables: Dict[int, List[Table]], output_dir: str, 
                    format: str = "csv") -> None:
        """Save extracted tables to files.

        Args:
            tables: Dictionary of page numbers to lists of tables.
            output_dir: Directory to save files.
            format: Output format ('csv' or 'json').
        """
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        for first_page, table_list in tables.items():
            for table_idx, table in enumerate(table_list):
                page_str = f"pages_{'_'.join(map(str, [p+1 for p in table.continuous_pages]))}"
                filename = f"table_{page_str}_table_{table_idx + 1}.{format}"
                filepath = Path(output_dir) / filename
                try:
                    if format == "csv":
                        table.to_csv(str(filepath))
                    elif format == "json":
                        table.to_json(str(filepath))
                    else:
                        raise ValueError(f"Unsupported format: {format}")
                    logger.info(f"Saved table to {filepath}")
                except Exception as e:
                    logger.error(f"Error saving table to {filepath}: {e}")
    
    def process(self, file_path: str, output_dir: Optional[str] = None, 
                page_numbers: Optional[List[int]] = None, strategy: Optional[str] = None, 
                output_format: str = "csv", password: Optional[str] = None) -> Dict[str, Any]:
        """Process a PDF to extract tables and text.

        Args:
            file_path: Path to the PDF file.
            output_dir: Directory to save output files.
            page_numbers: List of page numbers (0-based) to process.
            strategy: Table extraction strategy to override default.
            output_format: Format for saved tables ('csv' or 'json').
            password: Password for encrypted PDFs.

        Returns:
            Dictionary with extracted text and tables.
        """
        result = {}
        doc = self.open(file_path, password)
        try:
            result["text"] = self.extract_text(doc, page_numbers)
            result["tables"] = self.extract_tables(doc, page_numbers, strategy)
            if output_dir:
                self.save_tables(result["tables"], output_dir, output_format)
        finally:
            doc.close()
        return result