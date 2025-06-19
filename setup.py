from setuptools import setup, find_packages

setup(
    name="pdf_table_extractor",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "pymupdf>=1.24.0",
        "numpy>=1.24.0",
        "tqdm>=4.65.0",
        "opencv-python>=4.8.0",
    ],
    extras_require={
        "full": [
            "pandas>=2.0.0",
            "matplotlib>=3.7.0",
            "rtree>=1.0.0",
            "scikit-learn>=1.3.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "pdf-table-extract = pdf_table_extractor.cli:main",
        ],
    },
    author="Your Name",
    author_email="your.email@example.com",
    description="A library for extracting tables and text from PDFs",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/pdf_table_extractor",
    license="MIT",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
)