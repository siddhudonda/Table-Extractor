import numpy as np
import cv2
from typing import List, Tuple
from dataclasses import dataclass, field
import fitz
import logging

try:
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

try:
    from sklearn.cluster import DBSCAN
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

logger = logging.getLogger("PDFTableExtractor")

@dataclass
class TableCell:
    """Represents a cell in a table with its content and position."""
    text: str
    bbox: Tuple[float, float, float, float]
    row_idx: int
    col_idx: int
    rowspan: int = 1
    colspan: int = 1
    is_header: bool = False

@dataclass
class TableRow:
    """Represents a row in a table with its cells and position."""
    cells: List[TableCell]
    bbox: Tuple[float, float, float, float]
    is_header: bool = False

@dataclass
class Table:
    """Represents a table with rows, cells, and metadata."""
    rows: List[TableRow] = field(default_factory=list)
    bbox: Tuple[float, float, float, float] = (0, 0, 0, 0)
    page_number: int = 0
    header_rows: int = 0
    continuous_pages: List[int] = field(default_factory=list)
    is_landscape: bool = False
    
    @property
    def num_rows(self) -> int:
        """Number of rows in the table."""
        return len(self.rows)
    
    @property
    def num_cols(self) -> int:
        """Number of columns in the table."""
        return max(len(row.cells) for row in self.rows) if self.rows else 0
    
    def to_list(self) -> List[List[str]]:
        """Convert table to list of lists."""
        return [[cell.text for cell in row.cells] for row in self.rows]

class TableExtractor:
    """Base class for table extraction."""
    def __init__(self, debug: bool = False):
        self.debug = debug
    
    def extract(self, page: fitz.Page, rotation: float = 0) -> List[Table]:
        """Extract tables from a page."""
        raise NotImplementedError
    
    def _debug_show(self, img, rects=None, title="Debug"):
        if not self.debug or not MATPLOTLIB_AVAILABLE:
            return
        plt.figure(figsize=(12, 12))
        plt.imshow(img)
        if rects:
            ax = plt.gca()
            for rect in rects:
                x0, y0, x1, y1 = rect
                ax.add_patch(Rectangle((x0, y0), x1-x0, y1-y0, linewidth=1, edgecolor='r', facecolor='none'))
        plt.title(title)
        plt.axis('off')
        plt.show()

class MuPDFTableExtractor(TableExtractor):
    """Extract tables using PyMuPDF's built-in detection."""
    def __init__(self, debug: bool = False, use_clipping: bool = True):
        super().__init__(debug)
        self.use_clipping = use_clipping
    
    def extract(self, page: fitz.Page, rotation: float = 0) -> List[Table]:
        tables = []
        options = {"vertical_strategy": "lines", "horizontal_strategy": "lines"}
        if self.use_clipping:
            text_areas = page.get_text("words")
            if text_areas:
                coords = [(float(w[0]), float(w[1]), float(w[2]), float(w[3])) for w in text_areas]
                if rotation in [90, 270]:
                    coords = [(y0, page.rect.height-x1, y1, page.rect.height-x0) for x0, y0, x1, y1 in coords]
                x0 = min(x for x, _, _, _ in coords)
                y0 = min(y for _, y, _, _ in coords)
                x1 = max(x for _, _, x, _ in coords)
                y1 = max(y for _, _, _, y in coords)
                options["clip"] = (x0-5, y0-5, x1+5, y1+5)
        
        found_tables = page.find_tables(**options)
        for tab in found_tables:
            table = Table(bbox=tab.bbox, page_number=page.number, is_landscape=rotation != 0)
            if tab.header and tab.header.cells:
                header_row = TableRow(cells=[], bbox=tab.header.bbox, is_header=True)
                for idx, cell_bbox in enumerate(tab.header.cells):
                    if cell_bbox:
                        if rotation in [90, 270]:
                            x0, y0, x1, y1 = cell_bbox
                            cell_bbox = (y0, page.rect.height-x1, y1, page.rect.height-x0)
                        text = page.get_text("text", clip=cell_bbox).strip()
                        header_row.cells.append(TableCell(text=text, bbox=cell_bbox, row_idx=0, col_idx=idx, is_header=True))
                table.rows.append(header_row)
                table.header_rows = 1
            
            row_data = tab.extract()
            start_idx = 1 if table.header_rows > 0 else 0
            for row_idx, row in enumerate(row_data[start_idx:], start=start_idx):
                table_row = TableRow(cells=[], bbox=(0, 0, 0, 0))
                for col_idx, cell_text in enumerate(row):
                    cell_bbox = tab.cells[row_idx * tab.col_count + col_idx] if row_idx < len(tab.cells) else (0, 0, 0, 0)
                    if rotation in [90, 270]:
                        x0, y0, x1, y1 = cell_bbox
                        cell_bbox = (y0, page.rect.height-x1, y1, page.rect.height-x0)
                    table_row.cells.append(TableCell(
                        text=str(cell_text).strip(), bbox=cell_bbox, row_idx=row_idx, col_idx=col_idx
                    ))
                if table_row.cells:
                    table_row.bbox = (
                        min(c.bbox[0] for c in table_row.cells),
                        min(c.bbox[1] for c in table_row.cells),
                        max(c.bbox[2] for c in table_row.cells),
                        max(c.bbox[3] for c in table_row.cells)
                    )
                table.rows.append(table_row)
            table.continuous_pages = [page.number]
            tables.append(table)
        
        if self.debug:
            pix = page.get_pixmap(matrix=fitz.Matrix(1, 1))
            img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
            self._debug_show(img, [t.bbox for t in tables], f"MuPDF Tables (Page {page.number})")
        return tables

class GridBasedTableExtractor(TableExtractor):
    """Extract tables using grid line detection."""
    def __init__(self, debug: bool = False, min_cells: int = 4):
        super().__init__(debug)
        self.min_cells = min_cells
    
    def extract(self, page: fitz.Page, rotation: float = 0) -> List[Table]:
        tables = []
        zoom = 1.5
        pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom))
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
        gray = cv2.cvtColor(img, cv2.COLOR_RGBA2GRAY) if pix.n == 4 else cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
        
        scale = 30
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (img.shape[1]//scale, 1))
        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, img.shape[0]//scale))
        horizontal_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horizontal_kernel, iterations=1)
        vertical_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, vertical_kernel, iterations=1)
        grid = cv2.add(horizontal_lines, vertical_lines)
        
        contours, _ = cv2.findContours(grid, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return tables
        
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if w * h < 1000:
                continue
            bbox = (x/zoom, y/zoom, (x+w)/zoom, (y+h)/zoom)
            if rotation in [90, 270]:
                x0, y0, x1, y1 = bbox
                bbox = (y0, page.rect.height-x1, y1, page.rect.height-x0)
            table = Table(bbox=bbox, page_number=page.number, is_landscape=rotation != 0)
            
            text_blocks = page.get_text("words", clip=bbox)
            if not text_blocks:
                continue
            coords = [(float(b[0]), float(b[1]), float(b[2]), float(b[3])) for b in text_blocks]
            if rotation in [90, 270]:
                coords = [(y0, page.rect.height-x1, y1, page.rect.height-x0) for x0, y0, x1, y1 in coords]
            x_coords = sorted(set([x0 for x0, _, _, _ in coords] + [x1 for _, _, x1, _ in coords]))
            y_coords = sorted(set([y0 for _, y0, _, _ in coords] + [y1 for _, _, _, y1 in coords]))
            if len(x_coords) < 2 or len(y_coords) < 2:
                continue
            
            for row_idx in range(len(y_coords)-1):
                row = TableRow(cells=[], bbox=(bbox[0], y_coords[row_idx], bbox[2], y_coords[row_idx+1]), is_header=(row_idx==0))
                for col_idx in range(len(x_coords)-1):
                    cell_bbox = (x_coords[col_idx], y_coords[row_idx], x_coords[col_idx+1], y_coords[row_idx+1])
                    text = page.get_text("text", clip=cell_bbox).strip()
                    row.cells.append(TableCell(text=text, bbox=cell_bbox, row_idx=row_idx, col_idx=col_idx, is_header=(row_idx==0)))
                table.rows.append(row)
            
            if len(table.rows) > 1 and len(table.rows[0].cells) > 1 and sum(len(r.cells) for r in table.rows) >= self.min_cells:
                table.header_rows = 1
                table.continuous_pages = [page.number]
                tables.append(table)
        
        if self.debug:
            self._debug_show(img, [t.bbox for t in tables], f"Grid Tables (Page {page.number})")
        return tables

class TextBasedTableExtractor(TableExtractor):
    """Extract tables using text alignment and clustering."""
    def __init__(self, debug: bool = False, min_rows: int = 2, min_cols: int = 2):
        super().__init__(debug)
        self.min_rows = min_rows
        self.min_cols = min_cols
    
    def extract(self, page: fitz.Page, rotation: float = 0) -> List[Table]:
        tables = []
        blocks = page.get_text("dict")["blocks"]
        text_blocks = []
        for block in blocks:
            if block.get("type") == 0:
                for line in block.get("lines", []):
                    for span in line.get("spans", []):
                        text = span.get("text", "").strip()
                        if text:
                            x0, y0, x1, y1 = span["bbox"]
                            if rotation in [90, 270]:
                                x0, y0, x1, y1 = y0, page.rect.height-x1, y1, page.rect.height-x0
                            text_blocks.append({
                                "bbox": (x0, y0, x1, y1),
                                "text": text,
                                "font": span.get("font", ""),
                                "size": span.get("size", 0),
                                "flags": span.get("flags", 0)
                            })
        
        if len(text_blocks) < self.min_rows * self.min_cols:
            return tables
        
        if SKLEARN_AVAILABLE:
            coords = np.array([(b["bbox"][0] + b["bbox"][2]) / 2, (b["bbox"][1] + b["bbox"][3]) / 2] for b in text_blocks)
            avg_size = np.mean([b["size"] for b in text_blocks])
            db = DBSCAN(eps=avg_size * 2, min_samples=1).fit(coords)
            labels = db.labels_
            
            rows = {}
            for idx, label in enumerate(labels):
                if label != -1:
                    if label not in rows:
                        rows[label] = []
                    rows[label].append(text_blocks[idx])
            
            sorted_rows = sorted(
                [(label, min(b["bbox"][1] for b in blocks)) for label, blocks in rows.items()],
                key=lambda x: x[1]
            )
            row_blocks = [rows[label] for label, _ in sorted_rows]
            
            if len(row_blocks) < self.min_rows:
                return tables
            
            processed_rows = []
            for row in row_blocks:
                sorted_blocks = sorted(row, key=lambda b: b["bbox"][0])
                x_coords = np.array([(b["bbox"][0] + b["bbox"][2]) / 2 for b in sorted_blocks]).reshape(-1, 1)
                if len(x_coords) >= self.min_cols:
                    db_cols = DBSCAN(eps=avg_size * 1.5, min_samples=1).fit(x_coords)
                    col_labels = db_cols.labels_
                    cols = {}
                    for idx, label in enumerate(col_labels):
                        if label != -1:
                            if label not in cols:
                                cols[label] = []
                            cols[label].append(sorted_blocks[idx])
                    if len(cols) >= self.min_cols:
                        processed_rows.append([
                            cols[label] for label in sorted(cols.keys(), key=lambda l: min(b["bbox"][0] for b in cols[l]))
                        ])
            
            if len(processed_rows) < self.min_rows:
                return tables
            
            table = Table(
                bbox=(
                    min(b["bbox"][0] for r in row_blocks for b in r),
                    min(b["bbox"][1] for r in row_blocks for b in r),
                    max(b["bbox"][2] for r in row_blocks for b in r),
                    max(b["bbox"][3] for r in row_blocks for b in r)
                ),
                page_number=page.number,
                is_landscape=rotation != 0
            )
            
            first_row = processed_rows[0]
            other_rows = processed_rows[1:] if len(processed_rows) > 1 else []
            is_header = False
            if first_row:
                first_fonts = {(b["font"], b["flags"], b["size"]) for col in first_row for b in col}
                other_fonts = {(b["font"], b["flags"], b["size"]) for row in other_rows for col in row for b in col}
                is_header = len(first_fonts) == 1 and (not other_fonts or first_fonts != other_fonts)
            
            for row_idx, row_cols in enumerate(processed_rows):
                row_bbox = (
                    min(b["bbox"][0] for col in row_cols for b in col),
                    min(b["bbox"][1] for col in row_cols for b in col),
                    max(b["bbox"][2] for col in row_cols for b in col),
                    max(b["bbox"][3] for col in row_cols for b in col)
                )
                table_row = TableRow(cells=[], bbox=row_bbox, is_header=(row_idx == 0 and is_header))
                for col_idx, col_blocks in enumerate(row_cols):
                    cell_text = " ".join(b["text"] for b in sorted(col_blocks, key=lambda b: b["bbox"][0]))
                    cell_bbox = (
                        min(b["bbox"][0] for b in col_blocks),
                        min(b["bbox"][1] for b in col_blocks),
                        max(b["bbox"][2] for b in col_blocks),
                        max(b["bbox"][3] for b in col_blocks)
                    )
                    table_row.cells.append(TableCell(
                        text=cell_text, bbox=cell_bbox, row_idx=row_idx, col_idx=col_idx,
                        is_header=(row_idx == 0 and is_header)
                    ))
                table.rows.append(table_row)
            
            if len(table.rows) >= self.min_rows and all(len(r.cells) >= self.min_cols for r in table.rows):
                table.header_rows = 1 if is_header else 0
                table.continuous_pages = [page.number]
                tables.append(table)
        
        else:
            logger.warning("scikit-learn not installed; using fallback text extraction")
            y_sorted = sorted(text_blocks, key=lambda b: b["bbox"][1])
            rows = []
            current_row = [y_sorted[0]]
            y_threshold = y_sorted[0]["bbox"][1] + 2 * y_sorted[0]["size"]
            
            for block in y_sorted[1:]:
                if block["bbox"][1] <= y_threshold:
                    current_row.append(block)
                    y_threshold = max(b["bbox"][3] for b in current_row) + 2 * block["size"]
                else:
                    if len(current_row) >= self.min_cols:
                        rows.append(sorted(current_row, key=lambda b: b["bbox"][0]))
                    current_row = [block]
                    y_threshold = block["bbox"][1] + 2 * block["size"]
            
            if len(current_row) >= self.min_cols:
                rows.append(sorted(current_row, key=lambda b: b["bbox"][0]))
            
            if len(rows) < self.min_rows:
                return tables
            
            x_coords = sorted(set(
                b["bbox"][0] for row in rows for b in row
            ).union(
                b["bbox"][2] for row in rows for b in row
            ))
            if len(x_coords) <= self.min_cols:
                return tables
            
            table = Table(
                bbox=(
                    min(b["bbox"][0] for r in rows for b in r),
                    min(b["bbox"][1] for r in rows for b in r),
                    max(b["bbox"][2] for r in rows for b in r),
                    max(b["bbox"][3] for r in rows for b in r)
                ),
                page_number=page.number,
                is_landscape=rotation != 0
            )
            
            is_header = len(rows) > 0 and sum(b["size"] for b in rows[0]) / len(rows[0]) > (
                sum(b["size"] for r in rows[1:] for b in r) / (len(rows[1:]) * len(rows[0]) or 1)
            ) * 1.1
            
            for row_idx, row_blocks in enumerate(rows):
                row = TableRow(
                    cells=[],
                    bbox=(
                        min(b["bbox"][0] for b in row_blocks),
                        min(b["bbox"][1] for b in row_blocks),
                        max(b["bbox"][2] for b in row_blocks),
                        max(b["bbox"][3] for b in row_blocks)
                    ),
                    is_header=(row_idx == 0 and is_header)
                )
                for col_idx in range(len(x_coords)-1):
                    col_start, col_end = x_coords[col_idx], x_coords[col_idx+1]
                    col_blocks = [b for b in row_blocks if col_start <= (b["bbox"][0]+b["bbox"][2])/2 < col_end]
                    text = " ".join(b["text"] for b in sorted(col_blocks, key=lambda b: b["bbox"][0]))
                    cell_bbox = (
                        min((b["bbox"][0] for b in col_blocks), default=col_start),
                        min((b["bbox"][1] for b in col_blocks), default=row.bbox[1]),
                        max((b["bbox"][2] for b in col_blocks), default=col_end),
                        max((b["bbox"][3] for b in col_blocks), default=row.bbox[3])
                    )
                    row.cells.append(TableCell(
                        text=text, bbox=cell_bbox, row_idx=row_idx, col_idx=col_idx,
                        is_header=(row_idx == 0 and is_header)
                    ))
                table.rows.append(row)
            
            if len(table.rows) >= self.min_rows and all(len(r.cells) >= self.min_cols for r in table.rows):
                table.header_rows = 1 if is_header else 0
                table.continuous_pages = [page.number]
                tables.append(table)
        
        if self.debug:
            pix = page.get_pixmap(matrix=fitz.Matrix(1, 1))
            img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
            rects = [t.bbox for t in tables] + [r.bbox for t in tables for r in t.rows] + [c.bbox for t in tables for r in t.rows for c in r.cells]
            self._debug_show(img, rects, f"Text Tables (Page {page.number})")
        
        return tables