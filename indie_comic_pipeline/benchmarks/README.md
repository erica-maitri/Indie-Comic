# Benchmarks for indie_comic_pipeline

This directory contains scripts and documentation for benchmarking the pipeline against external comic datasets.

## Selected Datasets
- **Awesome Comic Dataset** – A curated collection of indie comic panels.
- **Manga109** – A Japanese manga dataset useful for layout evaluation.
- **Community GitHub Comics** – Various open‑source comic repositories on GitHub.

## Usage
1. Run `download_datasets.py` to fetch and preprocess the datasets.
2. Use `evaluate.py` to run the current pipeline on the benchmark set and generate quality metrics.

Feel free to add additional datasets by extending `download_datasets.py`.
