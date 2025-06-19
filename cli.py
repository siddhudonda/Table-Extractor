import argparse
from pdf_table_extractor.core import EnhancedPDFExtractor
from pdf_table_extractor.processor import PDFProcessor
from pdf_table_extractor.utils import validate_pdf_path, setup_logging

def main():
    parser = argparse.ArgumentParser(description="PDF Table and Text Extractor CLI")
    parser.add_argument("--input", required=True, help="Path to PDF file or directory")
    parser.add_argument("--output-dir", default="output", help="Output directory")
    parser.add_argument("--pages", type=str, help="Comma-separated page numbers (1-based)")
    parser.add_argument("--strategy", default="auto", 
                       choices=["auto", "grid", "text", "mupdf", "all"], help="Table extraction strategy")
    parser.add_argument("--format", default="csv", choices=["csv", "json"], help="Output format")
    parser.add_argument("--debug", action="store_true", help="Enable debug visualizations")
    parser.add_argument("--password", help="Password for encrypted PDFs")
    parser.add_argument("--batch", action="store_true", help="Process input as directory of PDFs")
    parser.add_argument("--continuous", action="store_true", default=True, 
                       help="Merge continuous tables across pages")
    parser.add_argument("--verbosity", default="INFO", 
                       choices=["DEBUG", "INFO", "WARNING", "ERROR"], help="Logging verbosity")
    
    args = parser.parse_args()
    setup_logging(args.verbosity)
    extractor = EnhancedPDFExtractor(debug=args.debug, strategy=args.strategy, continuous=args.continuous)
    processor = PDFProcessor(extractor, args.output_dir)
    
    page_numbers = [int(p)-1 for p in args.pages.split(",")] if args.pages else None
    
    if args.batch:
        if not Path(args.input).is_dir():
            logger.error(f"Directory does not exist: {args.input}")
            return
        results = processor.process_directory(
            input_dir=args.input,
            page_numbers=page_numbers,
            strategy=args.strategy,
            output_format=args.format,
            password=args.password
        )
        summary = processor.summarize_results(results)
        print("\nProcessing Summary:")
        print(f"Total Files: {summary['total_files']}")
        print(f"Successful: {summary['successful']}")
        print(f"Failed: {summary['failed']}")
        print(f"Total Tables: {summary['total_tables']}")
        print(f"Total Pages: {summary['total_pages']}")
    else:
        if not validate_pdf_path(args.input):
            return
        result = extractor.process(
            file_path=args.input,
            output_dir=args.output_dir,
            page_numbers=page_numbers,
            strategy=args.strategy,
            output_format=args.format,
            password=args.password
        )
        if "tables" in result:
            print("\nExtracted Tables:")
            for first_page, tables in result["tables"].items():
                for idx, table in enumerate(tables):
                    pages = ",".join(map(str, [p+1 for p in table.continuous_pages]))
                    print(f"Table {idx + 1} (Pages {pages}): {table.num_rows} rows x {table.num_cols} columns, "
                          f"{'Landscape' if table.is_landscape else 'Portrait'}")
        if "text" in result:
            print("\nExtracted Text:")
            for page_num, text_data in result["text"].items():
                print(f"Page {page_num + 1}:")
                print(text_data["raw"][:200] + "..." if len(text_data["raw"]) > 200 else text_data["raw"])

if __name__ == "__main__":
    main()